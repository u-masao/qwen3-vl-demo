"""テキスト→画像検索の精度を InformationRetrievalEvaluator で測定する。

評価の構図:
  * **クエリ (queries)**       … 各行のキャプション（テキスト）
  * **コーパス (corpus)**      … 生成した全画像の集合
  * **正解 (relevant_docs)**   … 各クエリは「自分自身のキャプションから作られた画像」に対応
                                 （``relevant_same_category`` が真なら同カテゴリ画像も正解に含める）

評価器はクエリ（テキスト）とコーパス（画像）をそれぞれ埋め込み、コサイン類似度で
ランキングして NDCG / Recall / MRR などを @k で算出する。結果は JSON に書き出すので、
ベースモデルとファインチューニング済みモデルの数値を後から比較できる。

このモジュールの ``build_ir_evaluator`` は train.py からも import され、学習中の
途中評価（evaluator コールバック）にも再利用される。
"""

from __future__ import annotations

import argparse
import json
import logging

from datasets import load_from_disk

from .config import (
    Config,
    add_common_args,
    add_config_args,
    add_embedding_args,
    config_from_args,
)
from .models import load_embedding_model
from .tracking import EXPERIMENT_NAME, Timer, cli_run, log_metrics

logger = logging.getLogger(__name__)

# 評価器の名前。出力されるメトリクスのキーにこの名前が前置される
# （例: "synthetic-image-retrieval_cosine_ndcg@10"）。app.py 側の表示と揃えてある。
EVALUATOR_NAME = "synthetic-image-retrieval"


def build_ir_evaluator(cfg: Config, name: str = EVALUATOR_NAME):
    """保存済みの eval スプリットから InformationRetrievalEvaluator を構築する。

    クエリはペルソナ名（"user_alpha" など）で、正解は同一ペルソナに属する全画像。
    視覚・テキストからは正解が推測できない嗜好ベースのマルチポジティブ検索タスク。
    """
    from sentence_transformers.evaluation import InformationRetrievalEvaluator

    eval_ds = load_from_disk(str(cfg.data_path / "eval"))

    queries: dict[str, str] = {}  # qid -> クエリ文（ペルソナ名）
    corpus: dict = {}  # cid -> 画像（PIL）
    relevant_docs: dict[str, set[str]] = {}  # qid -> 正解 cid の集合
    persona_to_cids: dict[str, set[str]] = {}  # ペルソナ -> その cid 集合

    # 1 周目: corpus / queries / persona_to_cids を構築する。
    for i, row in enumerate(eval_ds):
        qid = f"q{i}"
        cid = f"d{i}"
        queries[qid] = row["persona"]
        corpus[cid] = row["positive"]
        persona_to_cids.setdefault(row["persona"], set()).add(cid)

    # 2 周目: 同一ペルソナの全画像を正解集合とする（マルチポジティブ）。
    for i, row in enumerate(eval_ds):
        qid = f"q{i}"
        relevant_docs[qid] = set(persona_to_cids[row["persona"]])

    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name=name,
        show_progress_bar=False,
        write_csv=False,
    )


def evaluate_model(cfg: Config, model_id: str, label: str) -> dict:
    """``model_id`` をロードして評価器を回し、メトリクス dict を返す。

    メトリクスは ``<output_dir>/metrics_<label>.json`` にも書き出す。
    ``label`` は通常 "base"（ベース）/ "finetuned"（FT 後）を使う。
    """
    logger.info("評価 [%s]: %s", label, model_id)
    # 大まかな所要時間（モデルロード / 評価本体）を計測して MLflow に残す（処理速度の把握用）。
    with Timer() as t_load:
        model = load_embedding_model(cfg, model_id=model_id)
    evaluator = build_ir_evaluator(cfg)
    with Timer() as t_eval:
        metrics = evaluator(model)  # 評価器を呼ぶとメトリクス dict が返る
    log_metrics({"time.model_load_sec": t_load.elapsed, "time.eval_sec": t_eval.elapsed})

    cfg.output_path.mkdir(parents=True, exist_ok=True)
    out_file = cfg.output_path / f"metrics_{label}.json"
    with open(out_file, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, sort_keys=True)

    _print_headline(metrics, label)
    logger.info("  全メトリクス -> %s", out_file)
    return metrics


def _print_headline(metrics: dict, label: str) -> None:
    """主要メトリクスだけを抜き出して表示する（キー名は ST のバージョンで変わりうる）。"""
    interesting = ("ndcg@10", "recall@1", "recall@5", "recall@10", "mrr@10")
    found = {k: v for k, v in metrics.items() if any(k.lower().endswith(s) for s in interesting)}
    if not found:
        logger.info("  [%s] metrics: %s", label, metrics)
        return
    for k in sorted(found):
        logger.info("  [%s] %s: %.4f", label, k, found[k])


def main() -> None:
    """CLI エントリポイント: ``python -m qwen3vl_demo.evaluate``。"""
    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser(description="テキスト→画像検索の精度を評価する。")
    add_config_args(parser)
    add_common_args(parser)
    add_embedding_args(parser)
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="評価するモデル ID またはパス（既定: 設定のベース埋め込みモデル）。",
    )
    parser.add_argument(
        "--finetuned",
        action="store_true",
        help="cfg.model_path に保存された FT 済みモデルを評価する（label は 'finetuned' になる）。",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="出力メトリクスファイルのラベル（既定: --finetuned 指定時は 'finetuned'、他は 'base'）。",
    )
    args = parser.parse_args()
    cfg = config_from_args(args)

    # --finetuned が指定されたら FT 済みモデルを優先。それ以外は --model か設定値。
    if args.finetuned:
        model_id = str(cfg.model_path)
        label = args.label or "finetuned"
    else:
        model_id = args.model or cfg.embedding.model_id
        label = args.label or "base"

    # MLflow: Experiment "evaluate" に 1 run として記録する（Issue #9）。リランクの
    # Retriever 単体（rerank=none）と同じ土俵に並ぶよう、tags を揃えておく。run は CLI 全体
    # （モデルロード含む）を覆うよう、ここ（引数解決直後）で開いて終了直前に閉じる。
    tags = {
        "stage": "evaluate",
        "label": label,
        "embedding": "ft" if args.finetuned else "base",
        "reranker": "none",
        "variant": "retriever",
    }
    with cli_run(EXPERIMENT_NAME, label, args=args, cfg=cfg, tags=tags):
        metrics = evaluate_model(cfg, model_id=model_id, label=label)
        log_metrics(metrics)


if __name__ == "__main__":
    main()
