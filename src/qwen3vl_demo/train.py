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
    cli_run,
    log_gpu_memory_status,
    log_time,
    make_curve_callback,
)

logger = logging.getLogger(__name__)


def train(cfg: Config) -> None:
    """設定に従ってファインチューニングを実行し、モデルを保存する。

    MLflow 記録は呼び出し側（``main()`` の :func:`cli_run`）が CLI 全体に対して開く run に
    ぶら下がる（アクティブ run が無ければ各記録は no-op）。学習曲線は TrainerCallback で記録。
    """
    from sentence_transformers import (
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.losses import MultipleNegativesRankingLoss

    # データ準備とハードネガティブマイニングを学習モデルのロード前に行う。
    # マイニング用埋め込みモデル（2B）と学習モデル（2B）を同時に VRAM に乗せると
    # 16GB カードで枯渇するため、マイニング → 解放 → 学習モデルロードの順にする。
    train_ds = load_from_disk(str(cfg.data_path / "train"))
    # ペルソナ名をアンカーとして使う（嗜好ベース検索タスク）。
    # persona 列を anchor に昇格させ、MNRL が期待する (anchor, positive) の形式にする。
    train_ds = train_ds.remove_columns(["anchor", "subject", "category"])
    train_ds = train_ds.rename_column("persona", "anchor")

    if cfg.train.num_negatives > 0:
        # ハードネガティブマイニング（Issue #33）: 埋め込みモデルでコーパスをエンコードし、
        # 各クエリに類似度上位の画像を負例として選ぶ。mine_hard_negatives 内でモデルを
        # ロード→解放するため、この時点では学習モデルはまだ VRAM に乗っていない。
        from datasets import Dataset, Features, Value
        from datasets import Image as HFImage

        from .train_reranker import mine_hard_negatives

        anchors_list = [row["anchor"] for row in train_ds]
        images_list = [row["positive"] for row in train_ds]
        logger.info("ハードネガティブをマイニング中... (num_negatives=%d)", cfg.train.num_negatives)
        pairs = mine_hard_negatives(
            cfg, anchors_list, images_list, cfg.train.num_negatives, seed=cfg.seed
        )

        neg_map: dict[int, list[int]] = {}
        for q_idx, d_idx, label in pairs:
            if label == 0.0:
                neg_map.setdefault(q_idx, []).append(d_idx)

        # PIL Image は add_column で PyArrow に直接変換できないため、Dataset.from_dict で
        # HFImage Feature を明示して再構築する（train_reranker.py と同方針）。
        n = len(train_ds)
        new_data: dict = {"anchor": anchors_list, "positive": images_list}
        feat_dict: dict = {"anchor": Value("string"), "positive": HFImage()}
        for slot in range(cfg.train.num_negatives):
            # フォールバック: 負例が足りない場合は index 0 の画像で埋める（コーパスが極小のとき）。
            neg_col = [
                images_list[neg_map[i][slot]]
                if i in neg_map and slot < len(neg_map[i])
                else images_list[0]
                for i in range(n)
            ]
            new_data[f"negative_{slot}"] = neg_col
            feat_dict[f"negative_{slot}"] = HFImage()

        train_ds = Dataset.from_dict(new_data, features=Features(feat_dict))
    else:
        # MNRL が必要とするのは (anchor, positive) の 2 カラムだけ。補助列は落としておく。
        keep = [c for c in ["anchor", "positive"] if c in train_ds.column_names]
        train_ds = train_ds.select_columns(keep)

    model = load_embedding_model(cfg)

    # 勾配チェックポイント: 計算を一部やり直す代わりに activation を保持せず VRAM を節約する。
    # 16GB の GPU で 2B モデルを回すための重要な節約手段。バックボーンが対応していなければ無視。
    if cfg.train.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
        except Exception as exc:  # noqa: BLE001 - ベストエフォート（全バックボーンが対応とは限らない）
            logger.warning("  勾配チェックポイントを有効化できませんでした: %s", exc)

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
        weight_decay=cfg.train.weight_decay,
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
    # アクティブな run が無ければ no-op になるので常に付けてよい。
    callbacks = []
    if (curve_cb := make_curve_callback()) is not None:
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

    # run（System Metrics・全設定）は main() の cli_run が CLI 全体に対して開く。
    # ここでは学習本体の所要時間だけ計測してアクティブ run に記録する。
    with log_time("time.train_total_sec"):
        trainer.train()
    log_gpu_memory_status()  # VRAM ピークと共有メモリ退避(spill)の有無を記録

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
    # run は CLI 全体（モデルロード・データ準備・学習）を覆う。
    with cli_run(TRAIN_EXPERIMENT_NAME, "train", args=args, cfg=cfg, tags={"stage": "train"}):
        train(cfg)


if __name__ == "__main__":
    main()
