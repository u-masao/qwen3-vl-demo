"""知識蒸留で「速い側（student）＝埋め込み bi-encoder」を作る。

埋め込み FT（train.py）／リランカー FT（train_reranker.py）に続く 3 本目の学習で、
**student は常に埋め込み bi-encoder**。teacher を 2 つから選んで、その知識を内積で速い
埋め込みへ移す。

* ``teacher = reranker``（パターン A: cross→bi 蒸留）
    FT 済みリランカー（cross-encoder）が出す関連度スコアを teacher とし、各クエリの
    (positive, negative) ペアについて teacher のマージン ``s_pos - s_neg`` を、student
    の類似度差が再現するよう **MarginMSELoss** で学習する。preference タスクの
    非加法的な交互作用（＝リランカーの伸びしろ。preference.py 参照）を、内積で速い
    bi-encoder にどこまで移せるかを測る実験そのもの。``reranker.model_id`` が null
    （smoke 等）のときはスキップする（リランカー teacher が無いため）。

* ``teacher = oracle``（パターン B: 嗖好モデル → bi 蒸留）
    ``preference.py`` の連続 appeal を soft relevance に変換し、(persona, image) ペアの
    soft label として **CoSENTLoss** で学習する。teacher 推論コスト 0（モデル不要）で、
    埋め込みと ``preference_model.json`` だけで動くため CPU/smoke でも検証できる。

負例は train_reranker.py と同じ **ハードネガティブマイニング**（埋め込み類似度上位）で選ぶ。
リランカー teacher のスコアリングは埋め込み（マイニング）→ リランカー → student の順に
**1 モデルずつ** ロード/解放し、16GB に同居させない（rerank.py / train_reranker.py と同方針）。

蒸留済みモデルは ``cfg.distill_model_path`` に保存する。評価は evaluate.py を
``--model <dir> --label distilled`` で流用する（専用コードは持たない）。
"""

from __future__ import annotations

import argparse
import logging

from datasets import Dataset, Features, Value, load_from_disk
from datasets import Image as HFImage

from .config import (
    Config,
    add_common_args,
    add_config_args,
    add_distill_args,
    add_embedding_args,
    add_reranker_args,
    add_train_args,
    config_from_args,
)
from .evaluate import EVALUATOR_NAME, build_ir_evaluator
from .models import load_embedding_model
from .preference import fragments_to_attributes, load_model, relevance_score
from .tracking import (
    TRAIN_EXPERIMENT_NAME,
    cli_run,
    log_gpu_memory_status,
    log_time,
    make_curve_callback,
)
from .train_reranker import _free_model, mine_hard_negatives

logger = logging.getLogger(__name__)


# --- 純粋関数（GPU/モデル不要・単体テスト対象）------------------------------
def _resolve_student_id(cfg: Config) -> str:
    """student（蒸留先）の初期化元モデル ID／パスを解決する（純粋関数）。

    蒸留先アーキを複数パターン試すための分岐。``distill.student_model`` に従う:

    * ``None``／空 … ベース埋め込み ``cfg.embedding.model_id`` から（自己蒸留）。
    * ``"ft"`` … FT 済み埋め込み成果物 ``cfg.model_path`` から継続蒸留する。
    * その他 … 任意の HF ID／ローカルパスをそのまま使う（小型 cross-modal 埋め込みへ圧縮）。

    Returns:
        ``SentenceTransformer`` に渡せるモデル ID 文字列。
    """
    student = cfg.distill.student_model
    if not student:  # None または空文字 → 自己蒸留
        return cfg.embedding.model_id
    if student == "ft":
        return str(cfg.model_path)
    return student


def group_negatives(
    pairs: list[tuple[int, int, float]],
) -> list[tuple[int, int, list[int]]]:
    """``mine_hard_negatives`` の (q, doc, label) 列をクエリ単位に畳む（純粋関数）。

    Args:
        pairs: ``(query_idx, doc_idx, label)`` のリスト。label==1.0 が正例、0.0 が負例。

    Returns:
        ``(query_idx, positive_idx, [negative_idx, ...])`` のリスト（query_idx 昇順）。
        正例の無いクエリは除外する。
    """
    positives: dict[int, int] = {}
    negatives: dict[int, list[int]] = {}
    for q, doc, label in pairs:
        if label >= 0.5:
            positives[q] = doc
        else:
            negatives.setdefault(q, []).append(doc)
    return [(q, positives[q], negatives.get(q, [])) for q in sorted(positives)]


def build_margin_rows(
    grouped: list[tuple[int, int, list[int]]],
    scores: dict[tuple[int, int], float],
) -> list[tuple[int, int, int, float]]:
    """teacher スコアから (query, positive, negative, margin) の三つ組行を作る（純粋関数）。

    margin = ``scores[(q, pos)] - scores[(q, neg)]``。MarginMSELoss の正解ラベルになる。

    Args:
        grouped: :func:`group_negatives` の出力。
        scores: ``{(query_idx, doc_idx): teacher_score}``（pos / 各 neg ぶんが必要）。

    Returns:
        ``(query_idx, positive_idx, negative_idx, margin)`` のリスト。
    """
    rows: list[tuple[int, int, int, float]] = []
    for q, pos, negs in grouped:
        s_pos = scores[(q, pos)]
        for neg in negs:
            rows.append((q, pos, neg, s_pos - scores[(q, neg)]))
    return rows


def build_oracle_rows(
    grouped: list[tuple[int, int, list[int]]],
    personas: list[str],
    texts: list[str],
    model,
) -> list[tuple[int, int, float]]:
    """嗖好モデルから (query, doc, soft_relevance) のペア行を作る（純粋関数）。

    各クエリ（ペルソナ ``personas[q]``）について、正例＋負例の各画像 ``doc`` の潜在属性を
    プロンプト文 ``texts[doc]`` から復元し、そのペルソナがその画像をどれだけ好むかの
    soft relevance（``preference.relevance_score`` ＝ sigmoid(appeal/温度)）を計算する。

    Args:
        grouped: :func:`group_negatives` の出力。
        personas: 各行のクエリ側ペルソナ（長さ = データ件数）。
        texts: 各行のプロンプト文（属性復元に使う。長さ = データ件数）。
        model: :class:`preference.PreferenceModel`。

    Returns:
        ``(query_idx, doc_idx, soft_relevance)`` のリスト（CoSENTLoss の連続ラベル）。
    """
    rows: list[tuple[int, int, float]] = []
    for q, pos, negs in grouped:
        for doc in (pos, *negs):
            attrs = fragments_to_attributes(model, texts[doc])
            rows.append((q, doc, relevance_score(model, personas[q], attrs)))
    return rows


# --- teacher 別のデータセット構築 -------------------------------------------
def _teacher_reranker_scores(
    cfg: Config,
    personas: list[str],
    images: list,
    grouped: list[tuple[int, int, list[int]]],
) -> dict[tuple[int, int], float]:
    """FT 済み（無ければベース）リランカーで、必要な (query, doc) ペアをスコアリングする。

    必要なペアは「各クエリの正例＋負例」だけ。リランカーを 1 度ロードして一括 predict し、
    解放する（埋め込みマイニングの後・student 学習の前に挟むことで VRAM 同居を避ける）。
    """
    from sentence_transformers import CrossEncoder

    # 必要な (q, doc) ペアを重複なく集める。
    needed: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for q, pos, negs in grouped:
        for doc in (pos, *negs):
            if (q, doc) not in seen:
                seen.add((q, doc))
                needed.append((q, doc))

    # FT 済みリランカーがあれば優先（rerank.py と同じ選択方針）。
    reranker_id = (
        str(cfg.reranker_model_path) if cfg.reranker_model_path.exists() else cfg.reranker.model_id
    )
    logger.info("  teacher リランカー = %s（%d ペアを採点）", reranker_id, len(needed))

    ce_kwargs: dict = {"device": cfg.device}
    if cfg.reranker.max_pixels:
        ce_kwargs["processor_kwargs"] = {"max_pixels": cfg.reranker.max_pixels}
    reranker = CrossEncoder(reranker_id, **ce_kwargs)

    # 8000件規模ではVRAMがWSL2共有メモリへ退避しOOMになるため、チャンク単位で採点し
    # バッチ間でCUDAキャッシュを解放する（Issue #25）。
    _SCORE_CHUNK = 256
    raw_scores: list[float] = []
    for start in range(0, len(needed), _SCORE_CHUNK):
        chunk = [[personas[q], images[doc]] for q, doc in needed[start : start + _SCORE_CHUNK]]
        scores = reranker.predict(chunk, show_progress_bar=False)
        raw_scores.extend(float(s) for s in scores)
        if cfg.device != "cpu":
            import torch

            torch.cuda.empty_cache()
    raw = raw_scores
    _free_model(reranker)

    return {pair: float(score) for pair, score in zip(needed, raw, strict=True)}


def _build_distill_dataset(cfg: Config) -> tuple[Dataset, str]:
    """train スプリットから teacher に応じた蒸留データセットを作る。

    Returns:
        ``(dataset, loss_kind)``。``loss_kind`` は "margin"（MarginMSE）/ "cosent"（CoSENT）。
    """
    train_ds = load_from_disk(str(cfg.data_path / "train"))
    personas = [row["persona"] for row in train_ds]  # クエリ側＝ペルソナ名
    images = [row["positive"] for row in train_ds]  # 文書側＝画像
    texts = [row["anchor"] for row in train_ds]  # 属性復元用のプロンプト文

    # ハードネガティブマイニング（埋め込みをロード→解放）を teacher ロードより先に行う。
    pairs = mine_hard_negatives(cfg, personas, images, cfg.distill.num_negatives, seed=cfg.seed)
    grouped = group_negatives(pairs)

    if cfg.distill.teacher == "oracle":
        # パターン B: 嗖好モデル（正解の作り手）の soft relevance を CoSENT ラベルにする。
        model = load_model(cfg.data_path / "preference_model.json")
        rows = build_oracle_rows(grouped, personas, texts, model)
        features = Features(
            {"query": Value("string"), "answer": HFImage(), "label": Value("float32")}
        )
        data = {
            "query": [personas[q] for q, _, _ in rows],
            "answer": [images[doc] for _, doc, _ in rows],
            "label": [label for _, _, label in rows],
        }
        return Dataset.from_dict(data, features=features), "cosent"

    # パターン A: リランカー teacher のマージンを MarginMSE ラベルにする。
    scores = _teacher_reranker_scores(cfg, personas, images, grouped)
    rows = build_margin_rows(grouped, scores)
    features = Features(
        {
            "query": Value("string"),
            "positive": HFImage(),
            "negative": HFImage(),
            "label": Value("float32"),
        }
    )
    data = {
        "query": [personas[q] for q, _, _, _ in rows],
        "positive": [images[pos] for _, pos, _, _ in rows],
        "negative": [images[neg] for _, _, neg, _ in rows],
        "label": [margin for _, _, _, margin in rows],
    }
    return Dataset.from_dict(data, features=features), "margin"


def distill(cfg: Config) -> None:
    """設定に従って teacher → student（埋め込み）蒸留を実行し、モデルを保存する。

    MLflow 記録は呼び出し側（``main()`` の :func:`cli_run`）が CLI 全体に対して開く run に
    ぶら下がる（アクティブ run が無ければ各記録は no-op）。学習曲線は TrainerCallback で記録。
    """
    if cfg.distill.teacher not in ("reranker", "oracle"):
        raise ValueError(
            f"distill.teacher は 'reranker' か 'oracle' を指定してください: {cfg.distill.teacher!r}"
        )

    # Phase 1 は bi-encoder student（量子化なし）のみ対応。cross / 量子化は今後のフェーズ。
    if cfg.distill.student_kind != "bi":
        raise NotImplementedError(
            f"distill.student_kind={cfg.distill.student_kind!r} は未対応です"
            "（現状は 'bi' のみ。cross-encoder 圧縮は今後対応予定）。"
        )
    if cfg.distill.quantize != "none":
        raise NotImplementedError(
            f"distill.quantize={cfg.distill.quantize!r} は未対応です"
            "（現状は 'none' のみ。量子化自己蒸留(QLoRA)は今後対応予定）。"
        )

    # student='ft'（FT 継続蒸留）なのに FT 成果物が無いなら早期に失敗させる。
    if cfg.distill.student_model == "ft" and not cfg.model_path.exists():
        raise FileNotFoundError(
            f"distill.student_model='ft' ですが FT 済み埋め込みが見つかりません: {cfg.model_path}"
            "（先に train ステージを実行してください）。"
        )

    if cfg.distill.teacher == "reranker" and not cfg.reranker.model_id:
        # リランカー teacher が無い（smoke 等）ので何もしない（CI でも安全にスキップ）。
        logger.info(
            "リランカー teacher が無効（reranker.model_id が null）のため、蒸留をスキップします。"
        )
        return

    from sentence_transformers import (
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.losses import CoSENTLoss, MarginMSELoss

    # データ構築（マイニング・teacher 採点）を student のロードより先に行い、VRAM 同居を避ける。
    train_ds, loss_kind = _build_distill_dataset(cfg)

    # student の初期化元は distill.student_model で選べる（既定＝ベース埋め込みの自己蒸留）。
    student_id = _resolve_student_id(cfg)
    model = load_embedding_model(cfg, model_id=student_id)

    if cfg.train.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
        except Exception as exc:  # noqa: BLE001 - ベストエフォート（全バックボーン対応とは限らない）
            logger.warning("  勾配チェックポイントを有効化できませんでした: %s", exc)

    loss = MarginMSELoss(model) if loss_kind == "margin" else CoSENTLoss(model)
    evaluator = build_ir_evaluator(cfg)  # 学習中の途中評価（eval スプリット由来）

    use_bf16 = cfg.device != "cpu" and cfg.dtype == "bfloat16"
    use_fp16 = cfg.device != "cpu" and cfg.dtype == "float16"

    args = SentenceTransformerTrainingArguments(
        output_dir=str(cfg.output_path / "distill_checkpoints"),
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
        save_strategy="best",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model=f"{EVALUATOR_NAME}_cosine_ndcg@10",
        greater_is_better=True,
        logging_steps=cfg.train.logging_steps,
        report_to=[],
        seed=cfg.seed,
    )

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

    logger.info(
        "student=%s を teacher=%s（loss=%s）で %d 件に蒸留します",
        student_id,
        cfg.distill.teacher,
        loss_kind,
        len(train_ds),
    )

    with log_time("time.train_total_sec"):
        trainer.train()
    log_gpu_memory_status()  # VRAM ピークと共有メモリ退避(spill)の有無を記録

    cfg.distill_model_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(cfg.distill_model_path))
    logger.info("蒸留済み student 埋め込みを %s に保存しました", cfg.distill_model_path)


def main() -> None:
    """CLI エントリポイント: ``python -m qwen3vl_demo.distill``。"""
    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser(
        description="teacher（リランカー / 嗖好モデル）→ student 埋め込みの知識蒸留。"
    )
    add_config_args(parser)
    add_common_args(parser)
    add_embedding_args(parser)
    add_reranker_args(parser)
    add_distill_args(parser)
    add_train_args(parser)
    args = parser.parse_args()
    cfg = config_from_args(args)
    # run は CLI 全体（マイニング・teacher 採点・蒸留）を覆う。
    with cli_run(
        TRAIN_EXPERIMENT_NAME,
        "distill",
        args=args,
        cfg=cfg,
        tags={"stage": "distill", "teacher": cfg.distill.teacher},
    ):
        distill(cfg)


if __name__ == "__main__":
    main()
