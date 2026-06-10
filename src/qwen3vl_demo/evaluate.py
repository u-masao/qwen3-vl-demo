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

from .config import Config, add_config_args, config_from_args
from .models import load_embedding_model

logger = logging.getLogger(__name__)

# 評価器の名前。出力されるメトリクスのキーにこの名前が前置される
# （例: "synthetic-image-retrieval_cosine_ndcg@10"）。app.py 側の表示と揃えてある。
EVALUATOR_NAME = "synthetic-image-retrieval"


def build_ir_evaluator(cfg: Config, name: str = EVALUATOR_NAME):
    """保存済みの eval スプリットから InformationRetrievalEvaluator を構築する。

    各行 i に対してクエリ ``q{i}`` と文書 ``d{i}`` を割り当て、``q{i}`` の正解を
    ``d{i}``（同じ行の画像）とする厳密 1 対 1 のマッピングを基本とする。
    ``cfg.data.relevant_same_category`` が真なら、同一カテゴリの全文書も正解集合に加える。
    """
    from sentence_transformers.evaluation import InformationRetrievalEvaluator

    eval_ds = load_from_disk(str(cfg.data_path / "eval"))

    queries: dict[str, str] = {}  # qid -> クエリ文
    corpus: dict = {}  # cid -> 画像（PIL）
    relevant_docs: dict[str, set[str]] = {}  # qid -> 正解 cid の集合
    category_to_cids: dict[str, set[str]] = {}  # カテゴリ -> その cid 集合（緩い評価用）
    subject_to_cids: dict[str, set[str]] = {}  # 主語 -> その cid 集合（subject ベース評価用）

    # 1 周目: corpus / queries / subject_to_cids / category_to_cids を構築する。
    # relevant_docs は subject_to_cids が確定してから 2 周目で埋める。
    for i, row in enumerate(eval_ds):
        qid = f"q{i}"
        cid = f"d{i}"
        # クエリは主語単語のみ（"cat" など）: 視覚分類タスクとして評価するため。
        queries[qid] = row["subject"]
        corpus[cid] = row["positive"]  # Image 型カラムなので PIL 画像としてデコードされる
        category_to_cids.setdefault(row["category"], set()).add(cid)
        subject_to_cids.setdefault(row["subject"], set()).add(cid)

    # 2 周目: 同一主語の全画像を正解集合とする。
    for i, row in enumerate(eval_ds):
        qid = f"q{i}"
        relevant_docs[qid] = set(subject_to_cids[row["subject"]])

    if cfg.data.relevant_same_category:
        # 緩い評価: 同じカテゴリの画像もすべて正解に追加する。
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
    """``model_id`` をロードして評価器を回し、メトリクス dict を返す。

    メトリクスは ``<output_dir>/metrics_<label>.json`` にも書き出す。
    ``label`` は通常 "base"（ベース）/ "finetuned"（FT 後）を使う。
    """
    logger.info("評価 [%s]: %s", label, model_id)
    model = load_embedding_model(cfg, model_id=model_id)
    evaluator = build_ir_evaluator(cfg)
    metrics = evaluator(model)  # 評価器を呼ぶとメトリクス dict が返る

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
    evaluate_model(cfg, model_id=model_id, label=label)


if __name__ == "__main__":
    main()
