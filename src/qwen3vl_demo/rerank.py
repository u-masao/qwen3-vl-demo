"""2 段階検索（埋め込み → リランカー）を 6 パターンで評価＋デモする。

実運用の検索システムでよく使われる「retrieve-then-rerank」構成を再現する:

  * **第 1 段（retrieve）**: 埋め込みモデルで各クエリ（キャプション）に対し画像コーパスから
    上位 k 件を高速に取得する（クエリと文書を別々にベクトル化して内積）。
  * **第 2 段（rerank）**: その k 件だけをリランカー（cross-encoder）でクエリと 1 件ずつ
    突き合わせて精密にスコアリングし、並べ替える。

埋め込み（base / ft）× リランカー（base / ft / なし）の **6 パターン** で検索精度を評価する:

  1. base  + none   （オリジナル埋め込み ＋ リランクなし＝参考値）
  2. ft    + none   （FT 埋め込み       ＋ リランクなし＝参考値）
  3. base  + base   （オリジナル埋め込み ＋ オリジナルリランカー）
  4. ft    + base   （FT 埋め込み       ＋ オリジナルリランカー）
  5. base  + ft     （オリジナル埋め込み ＋ FT リランカー）
  6. ft    + ft     （FT 埋め込み       ＋ FT リランカー）

各パターンについて NDCG / Recall@k / MRR を計算し ``outputs/rerank_metrics.json`` に保存する。
さらに数クエリ分の「リランク前後の順位」を ``outputs/rerank_examples.json`` に書き出す。

VRAM 16GB に収めるため、モデルは **同時に 1 つだけ** ロードし、使い終えたら解放する。
``reranker.model_id`` が null（スモーク等）のときは工程全体をスキップする。
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os

from datasets import load_from_disk

from .config import (
    Config,
    add_common_args,
    add_config_args,
    add_data_args,
    add_embedding_args,
    add_reranker_args,
    config_from_args,
)
from .models import load_embedding_model

logger = logging.getLogger(__name__)


def _ensure_alloc_conf() -> None:
    """CUDA メモリ断片化対策の環境変数を（未設定なら）有効化する（Issue #11）。

    同一プロセスで 2B 級モデルを順にロード/解放するため、解放後の領域が断片化して
    次のモデルが 16GB に収まらず共有システムメモリへ退避（PCIe 経由で激遅化）しやすい。
    ``expandable_segments:True`` は解放領域を伸縮可能セグメントとして再利用しやすくし、
    断片化による「実メモリは足りるのに確保に失敗→退避」を緩和する。

    この変数は CUDA キャッシュアロケータの初期化時（＝最初の CUDA 確保時）に読まれるため、
    最初のモデルロードより前に呼ぶこと。ユーザーが明示設定済みなら尊重する。
    """
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _cuda_mem_mb() -> tuple[float, float] | None:
    """現在の (allocated, reserved) を MiB で返す。CUDA 非対応なら None。"""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        mib = 1024 * 1024
        return (torch.cuda.memory_allocated() / mib, torch.cuda.memory_reserved() / mib)
    except Exception:  # noqa: BLE001 - torch が無い/CPU でも無視してよい
        return None


def _log_mem(label: str) -> None:
    """VRAM 使用量をログ出力する（base/ft の非対称や断片化を診断するため。Issue #11）。"""
    mem = _cuda_mem_mb()
    if mem is not None:
        logger.info("    [VRAM] %-12s allocated=%6.0f MiB / reserved=%6.0f MiB", label, *mem)


def _free() -> None:
    """GPU メモリを回収する（gc + empty_cache）。

    重要: 呼び出し側で対象モデルへの参照を ``del`` してから呼ぶこと。Python では
    ヘルパへ引数で渡しても呼び出し側のローカル束縛は残るため、ヘルパ内で ``del`` しても
    モデルは生きたままで ``empty_cache()`` が空振りする（旧実装の不具合。次モデルの
    ロード時に前モデルの領域が解放されず断片化 → VRAM 枯渇の一因だった。Issue #11）。
    """
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - torch が無い/CPU でも無視してよい
        pass


def _build_relevant(eval_ds, relevant_same_category: bool) -> list[set[int]]:
    """各クエリ i に対する正解文書インデックス集合を作る（純粋関数）。

    同一ペルソナの全文書を正解とする（マルチポジティブ検索タスク）。
    ``relevant_same_category`` が真なら同一カテゴリも正解に追加する。
    """
    persona_to_idx: dict[str, set[int]] = {}
    for i, row in enumerate(eval_ds):
        persona_to_idx.setdefault(row["persona"], set()).add(i)

    relevant: list[set[int]] = [set(persona_to_idx[row["persona"]]) for row in eval_ds]

    if relevant_same_category:
        categories = [row["category"] for row in eval_ds]
        by_cat: dict[str, set[int]] = {}
        for i, c in enumerate(categories):
            by_cat.setdefault(c, set()).add(i)
        for i in range(len(eval_ds)):
            relevant[i] |= by_cat[categories[i]]

    return relevant


def _metrics_for(
    ranked_lists: list[list[int]],
    relevant_sets: list[set[int]],
    ks: list[int],
) -> dict[str, float]:
    """ランキング結果から Recall@k / NDCG@k / MRR を計算する（純粋関数）。

    Args:
        ranked_lists: 各クエリの、関連度降順に並んだ文書インデックス列。
        relevant_sets: 各クエリの正解文書インデックス集合。
        ks: Recall / NDCG を測る上位件数のリスト。

    Returns:
        ``{"recall@k": ..., "ndcg@k": ..., "mrr": ...}`` の dict。
    """
    n = max(1, len(ranked_lists))
    out: dict[str, float] = {}

    for k in ks:
        recall_sum = 0.0
        ndcg_sum = 0.0
        for ranked, rel in zip(ranked_lists, relevant_sets, strict=False):
            if not rel:
                continue
            topk = ranked[:k]
            hits = sum(1 for d in topk if d in rel)
            recall_sum += hits / len(rel)
            # 2 値関連度の DCG / IDCG。
            dcg = sum(1.0 / math.log2(pos + 1) for pos, d in enumerate(topk, start=1) if d in rel)
            ideal = sum(1.0 / math.log2(p + 1) for p in range(1, min(k, len(rel)) + 1))
            ndcg_sum += (dcg / ideal) if ideal > 0 else 0.0
        out[f"recall@{k}"] = recall_sum / n
        out[f"ndcg@{k}"] = ndcg_sum / n

    # MRR: 最初に現れた正解の逆順位（ランキング全体を対象）。
    mrr_sum = 0.0
    for ranked, rel in zip(ranked_lists, relevant_sets, strict=False):
        for pos, d in enumerate(ranked, start=1):
            if d in rel:
                mrr_sum += 1.0 / pos
                break
    out["mrr"] = mrr_sum / n
    return out


def _retrieve_topk(
    cfg: Config,
    model_id: str,
    queries: list[str],
    corpus_images: list,
    top_k: int,
) -> list[list[int]]:
    """埋め込みモデルで各クエリの上位 top_k 文書インデックスを取得する（取得後に解放）。"""
    model = load_embedding_model(cfg, model_id=model_id)
    corpus_emb = model.encode(corpus_images, convert_to_tensor=True, show_progress_bar=False)
    query_emb = model.encode(queries, convert_to_tensor=True, show_progress_bar=False)
    sim = model.similarity(query_emb, corpus_emb)  # [クエリ数, コーパス数]
    retrieved = [sim[q].topk(top_k).indices.tolist() for q in range(len(queries))]
    # 次モデルのために確実に解放する: ローカル参照を消してから empty_cache を呼ぶ。
    del model, corpus_emb, query_emb, sim
    _free()
    return retrieved


def _rerank_candidates(
    cfg: Config,
    reranker_id: str,
    queries: list[str],
    corpus_images: list,
    candidate_lists: dict[str, list[list[int]]],
) -> dict[str, list[list[int]]]:
    """1 つのリランカーで、複数の候補集合（埋め込み別）をまとめて並べ替える（後に解放）。

    リランカーのロードは重いので、与えられた全 candidate_lists（base/ft 埋め込み由来）を
    この 1 ロードで処理してから解放し、VRAM を節約する。

    VRAM 枯渇対策（Issue #11）として:

    * dtype を明示し、base/ft を同一バイト幅でロードする（dtype 非対称を排除）。
    * ``max_pixels`` で画像トークン上限を下げ、forward の活性化メモリを抑える（埋め込み側と
      同方針）。リランカーの既定プロセッサは上限が極端に大きく、評価画像をフル解像度で
      処理してしまうため。
    * ``use_cache=False`` を base/ft 双方に強制する。リランクは単発 forward でスコアを得る
      だけなので KV キャッシュは不要。FT モデルは学習由来で既に False だが、base も揃えて
      対称化し、無駄な KV 確保を防ぐ。
    * ロード/解放の前後で VRAM を記録し、base/ft の非対称や断片化を可視化する。
    """
    from sentence_transformers import CrossEncoder

    from .config import resolve_dtype

    _log_mem("ロード前")

    kwargs: dict = {"device": cfg.device, "config_kwargs": {"use_cache": False}}
    if cfg.device != "cpu":
        # 既定（dtype="auto"）でも config の bfloat16 を読むが、念のため明示して base/ft を揃える。
        kwargs["model_kwargs"] = {"dtype": resolve_dtype(cfg.dtype)}
    if cfg.reranker.max_pixels:
        kwargs["processor_kwargs"] = {"max_pixels": cfg.reranker.max_pixels}

    reranker = CrossEncoder(reranker_id, **kwargs)
    _log_mem("ロード後")

    reranked: dict[str, list[list[int]]] = {}
    for emb_name, candidates in candidate_lists.items():
        per_query: list[list[int]] = []
        for q, cand in enumerate(candidates):
            images = [corpus_images[j] for j in cand]
            ranked = reranker.rank(queries[q], images, top_k=len(cand))
            per_query.append([cand[r["corpus_id"]] for r in ranked])
        reranked[emb_name] = per_query

    # 推論中のピークを記録してから解放する（base/ft の差を診断する手がかり）。
    peak = _cuda_mem_mb()
    if peak is not None:
        try:
            import torch

            logger.info(
                "    [VRAM] ピーク      allocated=%6.0f MiB",
                torch.cuda.max_memory_allocated() / (1024 * 1024),
            )
            torch.cuda.reset_peak_memory_stats()
        except Exception:  # noqa: BLE001
            pass

    # 次のリランカーのために確実に解放する（ローカル参照を消してから empty_cache）。
    del reranker
    _free()
    _log_mem("解放後")
    return reranked


def _rank_of(target: int, ordered: list[int]) -> int | None:
    """``ordered`` における ``target`` の順位（1 始まり）。無ければ None。"""
    for pos, d in enumerate(ordered, start=1):
        if d == target:
            return pos
    return None


def run_rerank(cfg: Config, num_queries: int = 5) -> None:
    """6 パターンの 2 段階検索を評価し、メトリクスと事例を保存する。"""
    if not cfg.reranker.model_id:
        # スモーク等、リランカー未設定の場合は何もしない。
        logger.info("リランカーが無効（reranker.model_id が null）のためスキップします。")
        return

    # 最初のモデルロードより前に断片化対策を有効化する（Issue #11）。
    _ensure_alloc_conf()

    eval_ds = load_from_disk(str(cfg.data_path / "eval"))
    corpus_images = [row["positive"] for row in eval_ds]
    # ペルソナ名をクエリとして使う（嗜好ベース検索タスク）。
    queries = [row["persona"] for row in eval_ds]
    relevant = _build_relevant(eval_ds, cfg.data.relevant_same_category)

    top_k = min(cfg.reranker.top_k, len(corpus_images))
    ks = sorted({1, min(5, top_k), top_k})

    # 埋め込み・リランカーそれぞれ base と（あれば）ft を用意する。
    emb_variants: dict[str, str] = {"base": cfg.embedding.model_id}
    if cfg.model_path.exists():
        emb_variants["ft"] = str(cfg.model_path)

    rr_variants: dict[str, str] = {"base": cfg.reranker.model_id}
    if cfg.reranker_model_path.exists():
        rr_variants["ft"] = str(cfg.reranker_model_path)

    logger.info(
        "埋め込み: %s / リランカー: %s / top_k=%d", list(emb_variants), list(rr_variants), top_k
    )

    # --- 第 1 段: 各埋め込みで候補（top_k）を取得（モデルは 1 つずつロード/解放） ---
    candidates: dict[str, list[list[int]]] = {}
    for emb_name, emb_id in emb_variants.items():
        logger.info("  retrieve: 埋め込み=%s (%s)", emb_name, emb_id)
        candidates[emb_name] = _retrieve_topk(cfg, emb_id, queries, corpus_images, top_k)

    metrics: dict[str, dict[str, float]] = {}

    # 参考: リランクなし（埋め込み検索のみ）の指標も記録する。
    for emb_name, cand in candidates.items():
        metrics[f"embed={emb_name}+rerank=none"] = _metrics_for(cand, relevant, ks)

    # --- 第 2 段: 各リランカーで全埋め込みの候補を並べ替えて評価 ---
    reranked_by_rr: dict[str, dict[str, list[list[int]]]] = {}
    for rr_name, rr_id in rr_variants.items():
        logger.info("  rerank: リランカー=%s (%s)", rr_name, rr_id)
        reranked = _rerank_candidates(cfg, rr_id, queries, corpus_images, candidates)
        reranked_by_rr[rr_name] = reranked
        for emb_name, ranked in reranked.items():
            metrics[f"embed={emb_name}+rerank={rr_name}"] = _metrics_for(ranked, relevant, ks)

    # メトリクスを保存。
    cfg.output_path.mkdir(parents=True, exist_ok=True)
    metrics_file = cfg.output_path / "rerank_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, sort_keys=True)

    # サマリを表示（主要指標 NDCG@top_k）。
    logger.info("=== 6 パターン評価（NDCG@%d / MRR）===", top_k)
    for key in sorted(metrics):
        m = metrics[key]
        logger.info(
            "  %-28s NDCG@%d=%.4f  MRR=%.4f", key, top_k, m.get(f"ndcg@{top_k}", 0), m["mrr"]
        )
    logger.info("  メトリクス -> %s", metrics_file)

    # --- 事例: 最良の組（ft があれば ft）でリランク前後を数件保存 ---
    # rank 変動が面白い（リランク前に正解が top-k 圏外 or 順位が変わった）クエリを優先表示。
    best_emb = "ft" if "ft" in emb_variants else "base"
    best_rr = "ft" if "ft" in rr_variants else "base"
    examples = []
    seen_queries: set[str] = set()
    for qi in range(len(eval_ds)):
        query = queries[qi]
        if query in seen_queries:
            continue  # 同一ペルソナクエリは代表 1 件だけ表示
        seen_queries.add(query)
        rel = relevant[qi]
        before = candidates[best_emb][qi]
        after = reranked_by_rr[best_rr][best_emb][qi]
        # 正解集合の中で最も上位に来た文書の順位を「代表順位」とする。
        best_before = min((pos for pos, d in enumerate(before, 1) if d in rel), default=None)
        best_after = min((pos for pos, d in enumerate(after, 1) if d in rel), default=None)
        # top-k 内に正解が何件含まれるか。
        hits_before = sum(1 for d in before if d in rel)
        hits_after = sum(1 for d in after if d in rel)
        examples.append(
            {
                "query": query,
                "num_relevant": len(rel),
                "embedding": best_emb,
                "reranker": best_rr,
                "top_k": top_k,
                "best_rank_before_rerank": best_before,
                "best_rank_after_rerank": best_after,
                "hits_in_topk_before": hits_before,
                "hits_in_topk_after": hits_after,
            }
        )
        if len(examples) >= num_queries:
            break
    examples_file = cfg.output_path / "rerank_examples.json"
    with open(examples_file, "w", encoding="utf-8") as fh:
        json.dump(examples, fh, indent=2)
    logger.info("  リランク事例（%s+%s）-> %s", best_emb, best_rr, examples_file)


def main() -> None:
    """CLI エントリポイント: ``python -m qwen3vl_demo.rerank``。"""
    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser(
        description="埋め込み×リランカーの 6 パターン 2 段階検索評価。"
    )
    add_config_args(parser)
    add_common_args(parser)
    add_data_args(parser)
    add_embedding_args(parser)
    add_reranker_args(parser)
    parser.add_argument(
        "--num-queries",
        type=int,
        default=5,
        help="事例として保存する eval クエリの件数。",
    )
    args = parser.parse_args()
    cfg = config_from_args(args)
    run_rerank(cfg, num_queries=args.num_queries)


if __name__ == "__main__":
    main()
