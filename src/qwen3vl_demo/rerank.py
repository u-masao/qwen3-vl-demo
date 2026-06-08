"""2 段階検索（埋め込み → リランカー）を 4 パターンで評価＋デモする。

実運用の検索システムでよく使われる「retrieve-then-rerank」構成を再現する:

  * **第 1 段（retrieve）**: 埋め込みモデルで各クエリ（キャプション）に対し画像コーパスから
    上位 k 件を高速に取得する（クエリと文書を別々にベクトル化して内積）。
  * **第 2 段（rerank）**: その k 件だけをリランカー（cross-encoder）でクエリと 1 件ずつ
    突き合わせて精密にスコアリングし、並べ替える。

埋め込み・リランカーそれぞれに「オリジナル（base）」と「ファインチューニング済み（ft）」が
あるので、その直積 **4 パターン** で検索精度を評価する:

  1. base  + base   （オリジナル埋め込み ＋ オリジナルリランカー）
  2. ft    + base   （FT 埋め込み       ＋ オリジナルリランカー）
  3. base  + ft     （オリジナル埋め込み ＋ FT リランカー）
  4. ft    + ft     （FT 埋め込み       ＋ FT リランカー）

各パターンについて NDCG / Recall@k / MRR を計算し ``outputs/rerank_metrics.json`` に保存する。
さらに数クエリ分の「リランク前後の順位」を ``outputs/rerank_examples.json`` に書き出す。

VRAM 16GB に収めるため、モデルは **同時に 1 つだけ** ロードし、使い終えたら解放する。
``reranker.model_id`` が null（スモーク等）のときは工程全体をスキップする。
"""

from __future__ import annotations

import argparse
import json
import math

from datasets import load_from_disk

from .config import Config, add_config_args, config_from_args
from .models import load_embedding_model


def _free(model) -> None:
    """モデルを破棄して GPU メモリを解放する（次のモデルを 16GB に収めるため）。"""
    import gc

    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - torch が無い/CPU でも無視してよい
        pass


def _build_relevant(eval_ds, relevant_same_category: bool) -> list[set[int]]:
    """各クエリ i に対する正解文書インデックス集合を作る（純粋関数）。

    基本は厳密 1 対 1（クエリ i の正解は文書 i）。``relevant_same_category`` が真なら
    同一カテゴリの文書もすべて正解に含める（緩い評価）。
    """
    n = len(eval_ds)
    relevant: list[set[int]] = [{i} for i in range(n)]
    if relevant_same_category:
        categories = [row["category"] for row in eval_ds]
        by_cat: dict[str, set[int]] = {}
        for i, c in enumerate(categories):
            by_cat.setdefault(c, set()).add(i)
        for i in range(n):
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
    _free(model)
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
    """
    from sentence_transformers import CrossEncoder

    reranker = CrossEncoder(reranker_id, device=cfg.device)
    reranked: dict[str, list[list[int]]] = {}
    for emb_name, candidates in candidate_lists.items():
        per_query: list[list[int]] = []
        for q, cand in enumerate(candidates):
            images = [corpus_images[j] for j in cand]
            ranked = reranker.rank(queries[q], images, top_k=len(cand))
            per_query.append([cand[r["corpus_id"]] for r in ranked])
        reranked[emb_name] = per_query
    _free(reranker)
    return reranked


def _rank_of(target: int, ordered: list[int]) -> int | None:
    """``ordered`` における ``target`` の順位（1 始まり）。無ければ None。"""
    for pos, d in enumerate(ordered, start=1):
        if d == target:
            return pos
    return None


def run_rerank(cfg: Config, num_queries: int = 5) -> None:
    """4 パターンの 2 段階検索を評価し、メトリクスと事例を保存する。"""
    if not cfg.reranker.model_id:
        # スモーク等、リランカー未設定の場合は何もしない。
        print("リランカーが無効（reranker.model_id が null）のためスキップします。")
        return

    eval_ds = load_from_disk(str(cfg.data_path / "eval"))
    corpus_images = [row["positive"] for row in eval_ds]
    queries = [row["anchor"] for row in eval_ds]
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

    print(f"埋め込み: {list(emb_variants)} / リランカー: {list(rr_variants)} / top_k={top_k}")

    # --- 第 1 段: 各埋め込みで候補（top_k）を取得（モデルは 1 つずつロード/解放） ---
    candidates: dict[str, list[list[int]]] = {}
    for emb_name, emb_id in emb_variants.items():
        print(f"  retrieve: 埋め込み={emb_name} ({emb_id})")
        candidates[emb_name] = _retrieve_topk(cfg, emb_id, queries, corpus_images, top_k)

    metrics: dict[str, dict[str, float]] = {}

    # 参考: リランクなし（埋め込み検索のみ）の指標も記録する。
    for emb_name, cand in candidates.items():
        metrics[f"embed={emb_name}+rerank=none"] = _metrics_for(cand, relevant, ks)

    # --- 第 2 段: 各リランカーで全埋め込みの候補を並べ替えて評価 ---
    reranked_by_rr: dict[str, dict[str, list[list[int]]]] = {}
    for rr_name, rr_id in rr_variants.items():
        print(f"  rerank: リランカー={rr_name} ({rr_id})")
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
    print("=== 4 パターン評価（NDCG@%d / MRR）===" % top_k)
    for key in sorted(metrics):
        m = metrics[key]
        print(f"  {key:28s} NDCG@{top_k}={m.get(f'ndcg@{top_k}', 0):.4f}  MRR={m['mrr']:.4f}")
    print(f"  メトリクス -> {metrics_file}")

    # --- 事例: 最良の組（ft があれば ft）でリランク前後の順位を数件保存 ---
    best_emb = "ft" if "ft" in emb_variants else "base"
    best_rr = "ft" if "ft" in rr_variants else "base"
    examples = []
    n = min(num_queries, len(eval_ds))
    for qi in range(n):
        before = candidates[best_emb][qi]
        after = reranked_by_rr[best_rr][best_emb][qi]
        examples.append(
            {
                "query": queries[qi],
                "target": qi,
                "embedding": best_emb,
                "reranker": best_rr,
                "rank_before_rerank": _rank_of(qi, before),
                "rank_after_rerank": _rank_of(qi, after),
                "top_k": top_k,
            }
        )
    examples_file = cfg.output_path / "rerank_examples.json"
    with open(examples_file, "w", encoding="utf-8") as fh:
        json.dump(examples, fh, indent=2)
    print(f"  リランク事例（{best_emb}+{best_rr}）-> {examples_file}")


def main() -> None:
    """CLI エントリポイント: ``python -m qwen3vl_demo.rerank``。"""
    parser = argparse.ArgumentParser(description="埋め込み×リランカーの 4 パターン 2 段階検索評価。")
    add_config_args(parser)
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
