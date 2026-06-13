"""合成 (caption, image) ペアで埋め込みモデルをファインチューニングする。

損失には **MultipleNegativesRankingLoss (MNRL)** を使う。MNRL の考え方:

  * バッチ内の各キャプションについて、対応する画像（``positive``）を正例とする。
  * 同じバッチに入っている **他のすべての画像** を負例（in-batch negatives）として扱う。
  * 正例との類似度を上げ、負例との類似度を下げるように学習する。

このため、バッチサイズが大きいほど 1 サンプルあたりの負例が増え、学習の質が上がりやすい
（VRAM と要相談）。明示的に負例を用意しなくてよいのが MNRL の利点で、(anchor, positive)
ペアさえあれば対照学習できる。

学習中は evaluate.py と同じ InformationRetrievalEvaluator を evaluator として渡し、
検索精度の推移を記録する。学習後のモデルは ``cfg.model_path`` に保存する。
"""

from __future__ import annotations

import argparse
import contextlib
import logging

from datasets import load_from_disk

from .config import (
    Config,
    add_common_args,
    add_config_args,
    add_embedding_args,
    add_train_args,
    config_from_args,
)
from .evaluate import build_ir_evaluator
from .models import load_embedding_model
from .tracking import (
    TRAIN_EXPERIMENT_NAME,
    args_to_params,
    config_to_params,
    enable_system_metrics,
    log_time,
    make_curve_callback,
    start_run,
)

logger = logging.getLogger(__name__)


def train(cfg: Config, cli_args: argparse.Namespace | None = None) -> None:
    """設定に従ってファインチューニングを実行し、モデルを保存する。

    ``cli_args`` を渡すと MLflow Experiment ``"train"`` に run として記録する（学習曲線・
    System Metrics・所要時間・全設定）。None の場合（テスト等）は記録しない。
    """
    from sentence_transformers import (
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.losses import MultipleNegativesRankingLoss

    model = load_embedding_model(cfg)

    # 勾配チェックポイント: 計算を一部やり直す代わりに activation を保持せず VRAM を節約する。
    # 16GB の GPU で 2B モデルを回すための重要な節約手段。バックボーンが対応していなければ無視。
    if cfg.train.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
        except Exception as exc:  # noqa: BLE001 - ベストエフォート（全バックボーンが対応とは限らない）
            logger.warning("  勾配チェックポイントを有効化できませんでした: %s", exc)

    train_ds = load_from_disk(str(cfg.data_path / "train"))
    # ペルソナ名をアンカーとして使う（嗜好ベース検索タスク）。
    # persona 列を anchor に昇格させ、MNRL が期待する (anchor, positive) の形式にする。
    train_ds = train_ds.remove_columns(["anchor", "subject", "category"])
    train_ds = train_ds.rename_column("persona", "anchor")
    # MNRL が必要とするのは (anchor, positive) の 2 カラムだけ。補助列は落としておく。
    keep = [c for c in ("anchor", "positive") if c in train_ds.column_names]
    train_ds = train_ds.select_columns(keep)

    loss = MultipleNegativesRankingLoss(model)
    evaluator = build_ir_evaluator(cfg)  # 学習中の途中評価に使う（評価器は eval スプリット由来）

    # Ada では bf16、明示的に float16 指定なら fp16、CPU では混合精度なし（full precision）。
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
        save_total_limit=1,  # チェックポイントは最新 1 個だけ残す（ディスク節約）
        logging_steps=cfg.train.logging_steps,
        report_to=[],  # W&B 等の外部ロガーへは送らない
        seed=cfg.seed,
    )

    # 学習曲線（loss / eval 指標）を MLflow に step 付きで記録するコールバック（Issue #9）。
    callbacks = []
    if cli_args is not None and (curve_cb := make_curve_callback()) is not None:
        callbacks.append(curve_cb)

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        loss=loss,
        evaluator=evaluator,
        callbacks=callbacks,
    )

    logger.info("%s を %d ペアでファインチューニングします", cfg.embedding.model_id, len(train_ds))

    # MLflow: Experiment "train" に run として記録（学習曲線・System Metrics・所要時間・全設定）。
    run_ctx = contextlib.nullcontext()
    if cli_args is not None:
        enable_system_metrics()
        params = {**args_to_params(cli_args), **config_to_params(cfg)}
        run_ctx = start_run(
            run_name="train",
            params=params,
            tags={"stage": "train"},
            experiment=TRAIN_EXPERIMENT_NAME,
        )
    with run_ctx, log_time("time.train_total_sec"):
        trainer.train()

    cfg.model_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(cfg.model_path))
    logger.info("ファインチューニング済みモデルを %s に保存しました", cfg.model_path)


def main() -> None:
    """CLI エントリポイント: ``python -m qwen3vl_demo.train``。"""
    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser(
        description="Qwen3-VL 埋め込みモデルをファインチューニングする。"
    )
    add_config_args(parser)
    add_common_args(parser)
    add_embedding_args(parser)
    add_train_args(parser)
    args = parser.parse_args()
    cfg = config_from_args(args)
    train(cfg, cli_args=args)


if __name__ == "__main__":
    main()
