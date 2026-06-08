"""2 段階検索のデモ: 埋め込みで粗く絞る → リランカーで精密に並べ替える。

実運用の検索システムでよく使われる「retrieve-then-rerank」構成を再現する:

  * **第 1 段（retrieve）**: ファインチューニング済み（あれば）の埋め込みモデルで、
    各クエリ（キャプション）に対し画像コーパスから上位 k 件を高速に取得する。
  * **第 2 段（rerank）**: その k 件だけを Qwen3-VL リランカー（cross-encoder）で
    クエリと 1 件ずつ突き合わせて精密にスコアリングし、並べ替える。

埋め込みは「クエリと文書を別々にベクトル化して内積」なので速いが粗い。リランカーは
「クエリと文書をペアでまとめて入力」するので精度は高いが重い。両者を組み合わせ、
速い埋め込みで候補を絞ってから重いリランカーを少数にだけ適用する、というのが定石。

各サンプルクエリについて、正解画像が「リランク前」と「リランク後」で何位だったかを表示し、
JSON にも書き出す。``reranker.model_id`` が null（スモークなど）のときは自動でスキップする。
"""

from __future__ import annotations

import argparse
import json

from datasets import load_from_disk

from .config import Config, add_config_args, config_from_args
from .models import load_embedding_model


def _rank_of(target_cid: str, ordered_cids: list[str]) -> int | None:
    """``ordered_cids`` の中での ``target_cid`` の順位（1 始まり）を返す。無ければ None。"""
    for pos, cid in enumerate(ordered_cids, start=1):
        if cid == target_cid:
            return pos
    return None


def run_rerank(cfg: Config, num_queries: int = 5) -> None:
    """埋め込み検索 → リランクを実行し、前後の順位変化を表示・保存する。"""
    if not cfg.reranker.model_id:
        # スモークプロファイルなどリランカー未設定の場合は何もしない。
        print("リランカーが無効（reranker.model_id が null）のためスキップします。")
        return

    # FT 済みモデルがあればそれを、無ければベースモデルを第 1 段の検索に使う。
    embed_model_id = str(cfg.model_path) if cfg.model_path.exists() else cfg.embedding.model_id
    print(f"第 1 段の検索に使う埋め込みモデル: {embed_model_id}")
    embed_model = load_embedding_model(cfg, model_id=embed_model_id)

    eval_ds = load_from_disk(str(cfg.data_path / "eval"))
    corpus_images = [row["positive"] for row in eval_ds]  # 検索対象の画像コーパス
    queries = [row["anchor"] for row in eval_ds]          # クエリ（キャプション）
    cids = [f"d{i}" for i in range(len(eval_ds))]         # 文書 ID（行 i ↔ d{i} ↔ 正解）

    # 第 1 段: コーパスとクエリを一度だけ埋め込み、コサイン類似度でランク付けする。
    corpus_emb = embed_model.encode(corpus_images, convert_to_tensor=True, show_progress_bar=False)
    query_emb = embed_model.encode(queries, convert_to_tensor=True, show_progress_bar=False)
    sim = embed_model.similarity(query_emb, corpus_emb)  # 形状 [クエリ数, コーパス数]

    top_k = min(cfg.reranker.top_k, len(corpus_images))

    # ファインチューニング済みリランカー（train_reranker.py の出力）があればそれを使う。
    rerank_model_id = (
        str(cfg.reranker_model_path)
        if cfg.reranker_model_path.exists()
        else cfg.reranker.model_id
    )
    print(f"第 2 段のリランクに使うモデル: {rerank_model_id}")
    from sentence_transformers import CrossEncoder

    reranker = CrossEncoder(rerank_model_id, device=cfg.device)

    n = min(num_queries, len(eval_ds))
    examples = []
    for qi in range(n):
        scores = sim[qi]
        # 埋め込み類似度の上位 top_k を取得（スコア降順のコーパスインデックス）。
        retrieved = scores.topk(top_k).indices.tolist()
        retrieved_cids = [cids[j] for j in retrieved]
        target_cid = cids[qi]                       # このクエリの正解文書 ID
        rank_before = _rank_of(target_cid, retrieved_cids)  # リランク前の正解順位

        # 取得した候補だけをリランカーで並べ替える。
        candidate_images = [corpus_images[j] for j in retrieved]
        ranked = reranker.rank(queries[qi], candidate_images, top_k=top_k)
        # rank() が返す corpus_id は candidate_images 内のインデックスなので cid に戻す。
        reranked_cids = [retrieved_cids[r["corpus_id"]] for r in ranked]
        rank_after = _rank_of(target_cid, reranked_cids)    # リランク後の正解順位

        examples.append(
            {
                "query": queries[qi],
                "target": target_cid,
                "rank_before_rerank": rank_before,
                "rank_after_rerank": rank_after,
                "top_k": top_k,
            }
        )
        print(
            f"  q{qi}: '{queries[qi][:50]}' | 正解画像の順位 "
            f"{rank_before} -> {rank_after}（top-{top_k} 中）"
        )

    cfg.output_path.mkdir(parents=True, exist_ok=True)
    out_file = cfg.output_path / "rerank_examples.json"
    with open(out_file, "w", encoding="utf-8") as fh:
        json.dump(examples, fh, indent=2)
    print(f"リランク事例を保存しました -> {out_file}")


def main() -> None:
    """CLI エントリポイント: ``python -m qwen3vl_demo.rerank``。"""
    parser = argparse.ArgumentParser(description="埋め込み検索 + リランクのデモ。")
    add_config_args(parser)
    parser.add_argument(
        "--num-queries",
        type=int,
        default=5,
        help="リランクの様子を表示する eval クエリの件数。",
    )
    args = parser.parse_args()
    cfg = config_from_args(args)
    run_rerank(cfg, num_queries=args.num_queries)


if __name__ == "__main__":
    main()
