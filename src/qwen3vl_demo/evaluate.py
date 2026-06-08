"""Evaluate text->image retrieval quality with an InformationRetrievalEvaluator.

Queries are the captions; the corpus is the set of generated images; each query
is relevant to its own image (and, optionally, to same-category images).
Metrics (NDCG / Recall / MRR @k) are printed and written to JSON so the base
and fine-tuned models can be compared.
"""

from __future__ import annotations

import argparse
import json

from datasets import load_from_disk

from .config import Config, add_config_args, config_from_args
from .models import load_embedding_model

EVALUATOR_NAME = "synthetic-image-retrieval"


def build_ir_evaluator(cfg: Config, name: str = EVALUATOR_NAME):
    """Construct an InformationRetrievalEvaluator from the saved eval split."""
    from sentence_transformers.evaluation import InformationRetrievalEvaluator

    eval_ds = load_from_disk(str(cfg.data_path / "eval"))

    queries: dict[str, str] = {}
    corpus: dict = {}
    relevant_docs: dict[str, set[str]] = {}
    category_to_cids: dict[str, set[str]] = {}

    for i, row in enumerate(eval_ds):
        qid = f"q{i}"
        cid = f"d{i}"
        queries[qid] = row["anchor"]
        corpus[cid] = row["positive"]  # PIL image (decoded by the Image feature)
        relevant_docs[qid] = {cid}
        category_to_cids.setdefault(row["category"], set()).add(cid)

    if cfg.data.relevant_same_category:
        for i, row in enumerate(eval_ds):
            qid = f"q{i}"
            relevant_docs[qid] |= category_to_cids[row["category"]]

    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name=name,
        show_progress_bar=False,
        write_csv=False,
    )


def evaluate_model(cfg: Config, model_id: str, label: str) -> dict:
    """Load ``model_id`` and run the IR evaluator; return the metrics dict."""
    print(f"Evaluating [{label}]: {model_id}")
    model = load_embedding_model(cfg, model_id=model_id)
    evaluator = build_ir_evaluator(cfg)
    metrics = evaluator(model)

    cfg.output_path.mkdir(parents=True, exist_ok=True)
    out_file = cfg.output_path / f"metrics_{label}.json"
    with open(out_file, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, sort_keys=True)

    _print_headline(metrics, label)
    print(f"  full metrics -> {out_file}")
    return metrics


def _print_headline(metrics: dict, label: str) -> None:
    """Print the few headline metrics if present (keys vary by ST version)."""
    interesting = ("ndcg@10", "recall@1", "recall@5", "recall@10", "mrr@10")
    found = {
        k: v
        for k, v in metrics.items()
        if any(k.lower().endswith(s) for s in interesting)
    }
    if not found:
        print(f"  [{label}] metrics: {metrics}")
        return
    for k in sorted(found):
        print(f"  [{label}] {k}: {found[k]:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate text->image retrieval.")
    add_config_args(parser)
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model id or path to evaluate (default: base embedding model from config).",
    )
    parser.add_argument(
        "--finetuned",
        action="store_true",
        help="Evaluate the fine-tuned model saved at cfg.model_path (implies label 'finetuned').",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Label for the output metrics file (default: 'finetuned' with --finetuned, else 'base').",
    )
    args = parser.parse_args()
    cfg = config_from_args(args)

    if args.finetuned:
        model_id = str(cfg.model_path)
        label = args.label or "finetuned"
    else:
        model_id = args.model or cfg.embedding.model_id
        label = args.label or "base"
    evaluate_model(cfg, model_id=model_id, label=label)


if __name__ == "__main__":
    main()
