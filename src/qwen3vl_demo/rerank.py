"""Two-stage retrieval demo: embed -> retrieve top-k -> rerank.

Stage 1 uses the (fine-tuned, if available) embedding model to retrieve the
top-k images for each query caption. Stage 2 reorders those k candidates with
the Qwen3-VL cross-encoder reranker. For each sampled query we print where the
correct image ranks before vs after reranking, and dump the examples to JSON.

Skipped automatically when ``reranker.model_id`` is null (e.g. smoke profile).
"""

from __future__ import annotations

import argparse
import json

from datasets import load_from_disk

from .config import Config, add_config_args, config_from_args
from .models import load_embedding_model


def _rank_of(target_cid: str, ordered_cids: list[str]) -> int | None:
    """1-based rank of ``target_cid`` in ``ordered_cids`` (None if absent)."""
    for pos, cid in enumerate(ordered_cids, start=1):
        if cid == target_cid:
            return pos
    return None


def run_rerank(cfg: Config, num_queries: int = 5) -> None:
    if not cfg.reranker.model_id:
        print("Reranker disabled (reranker.model_id is null) — skipping rerank demo.")
        return

    # Prefer the fine-tuned model if it exists, else fall back to the base model.
    embed_model_id = str(cfg.model_path) if cfg.model_path.exists() else cfg.embedding.model_id
    print(f"Stage 1 retrieval with embedding model: {embed_model_id}")
    embed_model = load_embedding_model(cfg, model_id=embed_model_id)

    eval_ds = load_from_disk(str(cfg.data_path / "eval"))
    corpus_images = [row["positive"] for row in eval_ds]
    queries = [row["anchor"] for row in eval_ds]
    cids = [f"d{i}" for i in range(len(eval_ds))]

    # Stage 1: embed corpus + queries once, rank by cosine similarity.
    corpus_emb = embed_model.encode(corpus_images, convert_to_tensor=True, show_progress_bar=False)
    query_emb = embed_model.encode(queries, convert_to_tensor=True, show_progress_bar=False)
    sim = embed_model.similarity(query_emb, corpus_emb)  # [num_queries, num_corpus]

    top_k = min(cfg.reranker.top_k, len(corpus_images))

    print(f"Stage 2 reranking with: {cfg.reranker.model_id}")
    from sentence_transformers import CrossEncoder

    reranker = CrossEncoder(cfg.reranker.model_id, device=cfg.device)

    n = min(num_queries, len(eval_ds))
    examples = []
    for qi in range(n):
        scores = sim[qi]
        retrieved = scores.topk(top_k).indices.tolist()  # corpus indices, best first
        retrieved_cids = [cids[j] for j in retrieved]
        target_cid = cids[qi]
        rank_before = _rank_of(target_cid, retrieved_cids)

        # Rerank the retrieved candidates with the cross-encoder.
        candidate_images = [corpus_images[j] for j in retrieved]
        ranked = reranker.rank(queries[qi], candidate_images, top_k=top_k)
        reranked_cids = [retrieved_cids[r["corpus_id"]] for r in ranked]
        rank_after = _rank_of(target_cid, reranked_cids)

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
            f"  q{qi}: '{queries[qi][:50]}' | correct image rank "
            f"{rank_before} -> {rank_after} (of top-{top_k})"
        )

    cfg.output_path.mkdir(parents=True, exist_ok=True)
    out_file = cfg.output_path / "rerank_examples.json"
    with open(out_file, "w", encoding="utf-8") as fh:
        json.dump(examples, fh, indent=2)
    print(f"Saved rerank examples -> {out_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed + retrieve + rerank demo.")
    add_config_args(parser)
    parser.add_argument(
        "--num-queries",
        type=int,
        default=5,
        help="How many eval queries to demonstrate reranking on.",
    )
    args = parser.parse_args()
    cfg = config_from_args(args)
    run_rerank(cfg, num_queries=args.num_queries)


if __name__ == "__main__":
    main()
