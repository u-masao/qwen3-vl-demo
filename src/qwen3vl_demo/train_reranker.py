"""リランカー（Qwen3-VL-Reranker-2B）のファインチューニング。

埋め込みモデルの学習（train.py）とは別に、cross-encoder であるリランカーを
合成データで微調整する。埋め込みが「クエリと文書を別々にベクトル化」するのに対し、
cross-encoder は「クエリと文書をペアで入力して 1 つの関連度スコアを出す」ため、
学習には **正例（一致ペア）と負例（不一致ペア）の両方** が必要になる。

負例生成には **ハードネガティブマイニング** を用いる:

  * FT 済み埋め込みモデル（なければベース）でコーパス全体をエンコードし、
    各クエリとのコサイン類似度が高い上位候補（正例を除く）を負例に選ぶ。
  * 実運用の検索パイプラインでリランカーが直面する「埋め込み上位候補」と同じ
    分布の難しい負例を学習させることで、精度改善を狙う。

損失は ``BinaryCrossEntropyLoss``（各 (query, image) ペアを「関連あり/なし」の
2 値分類として学習）を用いる。

注意（実験的）:
    マルチモーダル cross-encoder の学習は新しい機能で、``sentence-transformers>=5.4`` の
    マルチモーダル対応に依存する。``reranker.model_id`` が null（スモーク等）の場合は
    リランカー学習はスキップされる。本番（GPU）での利用を想定。
"""

from __future__ import annotations

import argparse
import gc
import logging
import random

from datasets import Dataset, Features, Value, load_from_disk
from datasets import Image as HFImage

from .config import (
    Config,
    add_common_args,
    add_config_args,
    add_embedding_args,
    add_reranker_args,
    add_train_args,
    config_from_args,
)
from .tracking import (
    TRAIN_EXPERIMENT_NAME,
    cli_run,
    log_gpu_memory_status,
    log_time,
    make_curve_callback,
)

logger = logging.getLogger(__name__)


def build_pair_indices(
    categories: list[str],
    num_negatives: int,
    seed: int,
) -> list[tuple[int, int, float]]:
    """(query_idx, doc_idx, label) の学習ペアを決定的に生成する（純粋関数）。

    各クエリ i に対して:
      * 正例 ``(i, i, 1.0)`` を 1 件
      * 負例 ``(i, j, 0.0)`` を ``num_negatives`` 件（``j != i``）

    負例の ``j`` は、できるだけ ``i`` と **別カテゴリ** から選ぶ（明確に不一致な
    負例にするため）。別カテゴリが足りなければ、同カテゴリの別インデックスで補う。
    画像を一切触らずインデックスだけ扱うので単体テストしやすい。

    .. note::
        本関数はテスト用・フォールバック用に残している。
        本番学習では :func:`mine_hard_negatives` を使うこと。

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


def _free_model(model) -> None:
    """モデルを破棄して GPU メモリを解放する。"""
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def mine_hard_negatives(
    cfg: Config,
    anchors: list[str],
    images: list,
    num_negatives: int,
    seed: int,
) -> list[tuple[int, int, float]]:
    """埋め込み類似度上位の候補を負例に使うハードネガティブマイニング（純粋関数）。

    FT 済み埋め込みモデル（``cfg.model_path``）があればそれを、なければベースモデル
    （``cfg.embedding.model_id``）を使ってコーパスをエンコードし、各クエリと
    コサイン類似度が高い順に並べた上位候補から正例（自分自身）を除いた
    ``num_negatives`` 件を負例として選ぶ。

    これにより、実際の検索パイプラインがリランカーに渡す「埋め込み上位候補」と
    同じ分布の難しい負例を学習させることができる。

    上位 ``num_negatives`` 件に満たない場合（コーパスが小さいとき等）は
    ランダムサンプリングで補充する。

    Args:
        cfg: パイプライン設定（device / dtype / 埋め込みモデルパス参照）。
        anchors: テキストクエリのリスト（長さ = データ件数）。
        images: 画像コーパス（長さ = データ件数）。
        num_negatives: 1 正例あたりの負例数。
        seed: ランダム補充時のシード（再現性）。

    Returns:
        ``(query_idx, doc_idx, label)`` のリスト。
    """
    from .models import load_embedding_model

    # FT 済み埋め込みモデルがあればそれを使う（再現性のため、学習後の分布に合わせる）。
    model_id = str(cfg.model_path) if cfg.model_path.exists() else cfg.embedding.model_id
    logger.info("  ハード負例採掘: 埋め込みモデル = %s", model_id)

    model = load_embedding_model(cfg, model_id=model_id)
    corpus_emb = model.encode(images, convert_to_tensor=True, show_progress_bar=False)
    query_emb = model.encode(anchors, convert_to_tensor=True, show_progress_bar=False)
    # sim[i][j] = クエリ i と画像 j のコサイン類似度。
    sim = model.similarity(query_emb, corpus_emb)
    _free_model(model)

    n = len(anchors)
    rng = random.Random(seed)
    pairs: list[tuple[int, int, float]] = []

    for i in range(n):
        pairs.append((i, i, 1.0))  # 正例

        if n <= 1 or num_negatives <= 0:
            continue

        # 類似度降順で上位候補を取得し、正例（i 自身）を除外する。
        fetch_k = min(num_negatives + 1, n)
        top_indices = sim[i].topk(fetch_k).indices.tolist()
        hard_negs = [j for j in top_indices if j != i][:num_negatives]

        # コーパスが小さくて足りない場合はランダム補充。
        if len(hard_negs) < num_negatives:
            already = set(hard_negs) | {i}
            remaining = [j for j in range(n) if j not in already]
            rng.shuffle(remaining)
            hard_negs.extend(remaining[: num_negatives - len(hard_negs)])

        for j in hard_negs:
            pairs.append((i, j, 0.0))  # ハード負例

    return pairs


def _build_reranker_dataset(cfg: Config) -> Dataset:
    """train スプリットからリランカー学習用の (query, answer, label) データセットを作る。"""
    train_ds = load_from_disk(str(cfg.data_path / "train"))
    # ペルソナ名をクエリとして使う（嗜好ベース検索タスク）。
    anchors = [row["persona"] for row in train_ds]
    images = [row["positive"] for row in train_ds]

    pairs = mine_hard_negatives(cfg, anchors, images, cfg.reranker.num_negatives, seed=cfg.seed)

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
    """設定に従ってリランカーをファインチューニングし、保存する。

    MLflow 記録は呼び出し側（``main()`` の :func:`cli_run`）が CLI 全体に対して開く run に
    ぶら下がる（アクティブ run が無ければ各記録は no-op）。学習曲線は TrainerCallback で記録。
    """
    if not cfg.reranker.model_id:
        # スモーク等、リランカー未設定の場合は何もしない（CI でも安全にスキップ）。
        logger.info("リランカーが無効（reranker.model_id が null）のため、学習をスキップします。")
        return

    from sentence_transformers.cross_encoder import (
        CrossEncoder,
        CrossEncoderTrainer,
        CrossEncoderTrainingArguments,
    )
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

    # ハード負例マイニング（埋め込みモデルをロード→解放）を **リランカーのロードより先に** 行う。
    # 順序を逆にすると、マイニング中に埋め込み(2B)とリランカー(2B)が VRAM に同居してピークが
    # 膨らみ、16GB カードでは枯渇する（WSL2 では共有メモリへ退避して極端に遅くなる）。
    train_ds = _build_reranker_dataset(cfg)

    # 学習時も画像トークンを max_pixels で上限化し、活性化メモリ（と共有メモリ退避）を抑える。
    # これが無いと reranker はフル解像度で画像を処理し、16GB を超えて WSL2 の共有メモリへ
    # 退避→激遅化する（rerank 評価側の Issue #11 対処と同方針を学習側にも適用）。
    ce_kwargs: dict = {"device": cfg.device}
    if cfg.reranker.max_pixels:
        ce_kwargs["processor_kwargs"] = {"max_pixels": cfg.reranker.max_pixels}
    model = CrossEncoder(cfg.reranker.model_id, **ce_kwargs)
    loss = BinaryCrossEntropyLoss(model)

    # Ada では bf16、明示的に float16 指定なら fp16、CPU では混合精度なし。
    use_bf16 = cfg.device != "cpu" and cfg.dtype == "bfloat16"
    use_fp16 = cfg.device != "cpu" and cfg.dtype == "float16"
    # 勾配チェックポイントで活性化メモリを抑える（埋め込み学習と同じ設定）。これが無いと
    # cross-encoder（クエリ＋画像を結合する分シーケンスが長い）の活性化が 16GB を超えて
    # VRAM 枯渇 → 退避で激遅化する（実測 ~3→18 s/it）。
    use_gc = cfg.device != "cpu" and cfg.train.gradient_checkpointing

    args = CrossEncoderTrainingArguments(
        output_dir=str(cfg.output_path / "reranker_checkpoints"),
        num_train_epochs=cfg.train.epochs,
        per_device_train_batch_size=cfg.train.per_device_batch_size,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        learning_rate=cfg.train.learning_rate,
        warmup_ratio=cfg.train.warmup_ratio,
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=use_gc,
        gradient_checkpointing_kwargs={"use_reentrant": False} if use_gc else None,
        save_strategy="steps",
        save_steps=cfg.train.save_steps,
        save_total_limit=1,
        logging_steps=cfg.train.logging_steps,
        report_to=[],
        seed=cfg.seed,
    )

    # 学習曲線（loss / eval 指標）を MLflow に step 付きで記録するコールバック（Issue #9）。
    # アクティブな run が無ければ no-op になるので常に付けてよい。
    callbacks = []
    if (curve_cb := make_curve_callback()) is not None:
        callbacks.append(curve_cb)

    trainer = CrossEncoderTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        loss=loss,
        callbacks=callbacks,
    )

    logger.info(
        "%s を %d ペア（正例＋負例）でファインチューニングします",
        cfg.reranker.model_id,
        len(train_ds),
    )

    # run（System Metrics・全設定）は main() の cli_run が CLI 全体に対して開く。
    # ここでは学習本体の所要時間だけ計測してアクティブ run に記録する。
    with log_time("time.train_total_sec"):
        trainer.train()
    log_gpu_memory_status()  # VRAM ピークと共有メモリ退避(spill)の有無を記録

    cfg.reranker_model_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(cfg.reranker_model_path))
    logger.info("ファインチューニング済みリランカーを %s に保存しました", cfg.reranker_model_path)


def main() -> None:
    """CLI エントリポイント: ``python -m qwen3vl_demo.train_reranker``。"""
    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser(description="Qwen3-VL リランカーをファインチューニングする。")
    add_config_args(parser)
    add_common_args(parser)
    add_embedding_args(parser)
    add_reranker_args(parser)
    add_train_args(parser)
    args = parser.parse_args()
    cfg = config_from_args(args)
    # run は CLI 全体（負例マイニング・モデルロード・学習）を覆う。
    with cli_run(
        TRAIN_EXPERIMENT_NAME,
        "train_reranker",
        args=args,
        cfg=cfg,
        tags={"stage": "train_reranker"},
    ):
        train_reranker(cfg)


if __name__ == "__main__":
    main()
