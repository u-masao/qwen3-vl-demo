"""リランカー（Qwen3-VL-Reranker-2B）のファインチューニング。

埋め込みモデルの学習（train.py）とは別に、cross-encoder であるリランカーを
合成データで微調整する。埋め込みが「クエリと文書を別々にベクトル化」するのに対し、
cross-encoder は「クエリと文書をペアで入力して 1 つの関連度スコアを出す」ため、
学習には **正例（一致ペア）と負例（不一致ペア）の両方** が必要になる。

このモジュールは合成データから負例を自動生成（ネガティブマイニング）して学習する:

  * 各キャプション i について、対応する画像 i を **正例（label=1）**
  * 同じキャプション i に対し、別の画像 j（可能なら別カテゴリ）を **負例（label=0）**

損失は ``BinaryCrossEntropyLoss``（各 (query, image) ペアを「関連あり/なし」の
2 値分類として学習）を用いる。

注意（実験的）:
    マルチモーダル cross-encoder の学習は新しい機能で、``sentence-transformers>=5.4`` の
    マルチモーダル対応に依存する。``reranker.model_id`` が null（スモーク等）の場合は
    リランカー学習はスキップされる。本番（GPU）での利用を想定。
"""

from __future__ import annotations

import argparse
import random

from datasets import Dataset, Features, Value, load_from_disk
from datasets import Image as HFImage

from .config import Config, add_config_args, config_from_args


def build_pair_indices(
    categories: list[str],
    num_negatives: int,
    seed: int,
) -> list[tuple[int, int, float]]:
    """(query_idx, doc_idx, label) の学習ペアを決定的に生成する（純粋関数）。

    各クエリ i に対して:
      * 正例 ``(i, i, 1.0)`` を 1 件
      * 負例 ``(i, j, 0.0)`` を ``num_negatives`` 件（``j != i``）

    負例の ``j`` は、できるだけ ``i`` と **別カテゴリ** から選ぶ（紛らわしすぎない、
    かつ明確に不一致な負例にするため）。別カテゴリが足りなければ、同カテゴリの
    別インデックスで補う。画像を一切触らずインデックスだけ扱うので単体テストしやすい。

    Args:
        categories: 各行の被写体カテゴリ（長さ = データ件数）。
        num_negatives: 1 正例あたりの負例数。
        seed: 乱数シード（再現性）。

    Returns:
        ``(query_idx, doc_idx, label)`` のリスト。
    """
    rng = random.Random(seed)
    n = len(categories)
    pairs: list[tuple[int, int, float]] = []

    for i in range(n):
        pairs.append((i, i, 1.0))  # 正例

        if n <= 1 or num_negatives <= 0:
            continue

        # 候補を「別カテゴリ優先、足りなければ同カテゴリ（自分以外）」の順で並べる。
        diff_cat = [j for j in range(n) if j != i and categories[j] != categories[i]]
        same_cat = [j for j in range(n) if j != i and categories[j] == categories[i]]
        rng.shuffle(diff_cat)
        rng.shuffle(same_cat)
        candidates = diff_cat + same_cat

        for j in candidates[:num_negatives]:
            pairs.append((i, j, 0.0))  # 負例

    return pairs


def _build_reranker_dataset(cfg: Config) -> Dataset:
    """train スプリットからリランカー学習用の (query, answer, label) データセットを作る。"""
    train_ds = load_from_disk(str(cfg.data_path / "train"))
    anchors = [row["anchor"] for row in train_ds]
    images = [row["positive"] for row in train_ds]
    categories = [row["category"] for row in train_ds]

    pairs = build_pair_indices(categories, cfg.reranker.num_negatives, seed=cfg.seed)

    features = Features(
        {
            "query": Value("string"),
            "answer": HFImage(),
            "label": Value("float32"),
        }
    )
    data = {
        "query": [anchors[q] for q, _, _ in pairs],
        "answer": [images[d] for _, d, _ in pairs],
        "label": [label for _, _, label in pairs],
    }
    return Dataset.from_dict(data, features=features)


def train_reranker(cfg: Config) -> None:
    """設定に従ってリランカーをファインチューニングし、保存する。"""
    if not cfg.reranker.model_id:
        # スモーク等、リランカー未設定の場合は何もしない（CI でも安全にスキップ）。
        print("リランカーが無効（reranker.model_id が null）のため、学習をスキップします。")
        return

    from sentence_transformers.cross_encoder import (
        CrossEncoder,
        CrossEncoderTrainer,
        CrossEncoderTrainingArguments,
    )
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

    model = CrossEncoder(cfg.reranker.model_id, device=cfg.device)

    train_ds = _build_reranker_dataset(cfg)
    loss = BinaryCrossEntropyLoss(model)

    # Ada では bf16、明示的に float16 指定なら fp16、CPU では混合精度なし。
    use_bf16 = cfg.device != "cpu" and cfg.dtype == "bfloat16"
    use_fp16 = cfg.device != "cpu" and cfg.dtype == "float16"

    args = CrossEncoderTrainingArguments(
        output_dir=str(cfg.output_path / "reranker_checkpoints"),
        num_train_epochs=cfg.train.epochs,
        per_device_train_batch_size=cfg.train.per_device_batch_size,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        learning_rate=cfg.train.learning_rate,
        warmup_ratio=cfg.train.warmup_ratio,
        bf16=use_bf16,
        fp16=use_fp16,
        save_strategy="steps",
        save_steps=cfg.train.save_steps,
        save_total_limit=1,
        logging_steps=cfg.train.logging_steps,
        report_to=[],
        seed=cfg.seed,
    )

    trainer = CrossEncoderTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        loss=loss,
    )

    print(f"{cfg.reranker.model_id} を {len(train_ds)} ペア（正例＋負例）でファインチューニングします")
    trainer.train()

    cfg.reranker_model_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(cfg.reranker_model_path))
    print(f"ファインチューニング済みリランカーを {cfg.reranker_model_path} に保存しました")


def main() -> None:
    """CLI エントリポイント: ``python -m qwen3vl_demo.train_reranker``。"""
    parser = argparse.ArgumentParser(description="Qwen3-VL リランカーをファインチューニングする。")
    add_config_args(parser)
    args = parser.parse_args()
    cfg = config_from_args(args)
    train_reranker(cfg)


if __name__ == "__main__":
    main()
