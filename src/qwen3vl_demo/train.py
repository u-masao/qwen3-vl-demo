"""Fine-tune the embedding model on the synthetic (caption, image) pairs.

Uses MultipleNegativesRankingLoss: within each batch, a caption's own image is
the positive and every other image is treated as an in-batch negative. The same
InformationRetrievalEvaluator from ``evaluate.py`` tracks progress during
training. The fine-tuned model is saved to ``cfg.model_path``.
"""

from __future__ import annotations

import argparse

from datasets import load_from_disk

from .config import Config, add_config_args, config_from_args
from .evaluate import build_ir_evaluator
from .models import load_embedding_model


def train(cfg: Config) -> None:
    from sentence_transformers import (
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.losses import MultipleNegativesRankingLoss

    model = load_embedding_model(cfg)

    if cfg.train.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
        except Exception as exc:  # noqa: BLE001 - best effort, not all backbones support it
            print(f"  gradient checkpointing not enabled: {exc}")

    train_ds = load_from_disk(str(cfg.data_path / "train"))
    # The loss only needs (anchor, positive); drop helper columns to be safe.
    keep = [c for c in ("anchor", "positive") if c in train_ds.column_names]
    train_ds = train_ds.select_columns(keep)

    loss = MultipleNegativesRankingLoss(model)
    evaluator = build_ir_evaluator(cfg)

    # bf16 on Ada, fp16 if explicitly requested, otherwise full precision (CPU).
    use_bf16 = cfg.device != "cpu" and cfg.dtype == "bfloat16"
    use_fp16 = cfg.device != "cpu" and cfg.dtype == "float16"

    args = SentenceTransformerTrainingArguments(
        output_dir=str(cfg.output_path / "checkpoints"),
        num_train_epochs=cfg.train.epochs,
        per_device_train_batch_size=cfg.train.per_device_batch_size,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        learning_rate=cfg.train.learning_rate,
        warmup_ratio=cfg.train.warmup_ratio,
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=cfg.train.gradient_checkpointing,
        eval_strategy="steps",
        eval_steps=cfg.train.eval_steps,
        save_strategy="steps",
        save_steps=cfg.train.save_steps,
        save_total_limit=1,
        logging_steps=cfg.train.logging_steps,
        report_to=[],
        seed=cfg.seed,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        loss=loss,
        evaluator=evaluator,
    )

    print(f"Fine-tuning {cfg.embedding.model_id} on {len(train_ds)} pairs")
    trainer.train()

    cfg.model_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(cfg.model_path))
    print(f"Saved fine-tuned model to {cfg.model_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune the Qwen3-VL embedding model.")
    add_config_args(parser)
    args = parser.parse_args()
    cfg = config_from_args(args)
    train(cfg)


if __name__ == "__main__":
    main()
