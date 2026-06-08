"""Qwen3-VL ファインチューニング・デモの結果ビューア（Gradio）。

パイプライン（generate_data → evaluate → train → eval → rerank）が出力した
成果物を、ブラウザ上で確認するための GUI。学習を回す機能はなく、あくまで
既に生成済みの ``data*/`` と ``outputs*/`` を読んで可視化するだけの読み取り専用ツール。

タブ構成:
  1. メトリクス比較   – ベース vs ファインチューニング後の棒グラフ＋数値表
  2. データセット閲覧  – 生成したキャプション付き画像を 1 枚ずつブラウズ
  3. Reranking デモ   – rerank_examples.json からリランク前後の順位変化を表示

起動: ``uv run python app.py`` → http://localhost:7860
"""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# サーバ環境（GUI ディスプレイ無し）で描画するため、非対話の Agg バックエンドを使う。
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# パス定義
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent

# 出力ディレクトリの選択肢。フル実行（outputs）とスモーク実行（outputs_smoke）を切替可能。
OUTPUT_DIRS = {
    "outputs (full run)": ROOT / "outputs",
    "outputs_smoke (smoke run)": ROOT / "outputs_smoke",
}

# データディレクトリの選択肢（同上）。
DATA_DIRS = {
    "data (full run)": ROOT / "data",
    "data_smoke (smoke run)": ROOT / "data_smoke",
}

# 表示対象とするメトリクス（表示順）。evaluate.py / 評価器が出すキーのうち主要なもの。
KEY_METRICS = [
    "accuracy@1",
    "accuracy@3",
    "accuracy@5",
    "accuracy@10",
    "recall@1",
    "recall@3",
    "recall@5",
    "recall@10",
    "ndcg@10",
    "mrr@10",
    "map@100",
]

# 評価器が付けるメトリクスキーの接頭辞（evaluate.py の EVALUATOR_NAME と対応）。
# 例: "synthetic-image-retrieval_cosine_ndcg@10" → 表示用に "ndcg@10" へ短縮する。
_PREFIX = "synthetic-image-retrieval_cosine_"


def _strip_prefix(key: str) -> str:
    """メトリクスキーから接頭辞を除いて短い表示名にする。"""
    return key[len(_PREFIX):] if key.startswith(_PREFIX) else key


# ---------------------------------------------------------------------------
# タブ 1（メトリクス比較）用のヘルパ
# ---------------------------------------------------------------------------

def load_metrics(output_dir_label: str) -> tuple[dict, dict]:
    """選択された出力ディレクトリから、ベース／FT 後のメトリクス JSON を読み込む。

    ファイルが無い場合は空 dict を返す（まだ評価していない場合などに備える）。
    """
    out_dir = OUTPUT_DIRS[output_dir_label]
    base_path = out_dir / "metrics_base.json"
    ft_path = out_dir / "metrics_finetuned.json"

    base = json.loads(base_path.read_text()) if base_path.exists() else {}
    ft = json.loads(ft_path.read_text()) if ft_path.exists() else {}
    return base, ft


def make_metrics_figure(output_dir_label: str):
    """ベース vs ファインチューニング後を並べた棒グラフ（matplotlib Figure）を作る。"""
    base, ft = load_metrics(output_dir_label)

    # KEY_METRICS のうち、どちらかのファイルに存在する項目だけを採用する。
    labels, base_vals, ft_vals = [], [], []
    for short_key in KEY_METRICS:
        full_key = _PREFIX + short_key
        if full_key in base or full_key in ft:
            labels.append(short_key)
            base_vals.append(base.get(full_key, 0.0))
            ft_vals.append(ft.get(full_key, 0.0))

    # メトリクスが 1 つも無ければ、その旨を描いた Figure を返す。
    if not labels:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "メトリクスデータが見つかりません", ha="center", va="center")
        return fig

    x = np.arange(len(labels))
    width = 0.35  # 棒の幅（ベースと FT を左右にずらして並べる）

    fig, ax = plt.subplots(figsize=(12, 5))
    bars_base = ax.bar(x - width / 2, base_vals, width, label="Base", color="#4C72B0", alpha=0.85)
    bars_ft = ax.bar(x + width / 2, ft_vals, width, label="Fine-tuned", color="#DD8452", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 1.12)  # スコアは 0〜1。注釈ラベル用に上を少し余らせる。
    ax.set_ylabel("Score")
    ax.set_title(f"Base vs Fine-tuned — {output_dir_label}", fontsize=11, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    def _annotate(bars):
        """各棒の上に数値ラベルを描く内部ヘルパ。"""
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(
                    f"{h:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    _annotate(bars_base)
    _annotate(bars_ft)

    fig.tight_layout()
    return fig


def make_metrics_table(output_dir_label: str) -> list[list]:
    """ベース／FT 後／差分（Δ）を並べた数値表（行のリスト）を作る。"""
    base, ft = load_metrics(output_dir_label)
    rows = []
    for short_key in KEY_METRICS:
        full_key = _PREFIX + short_key
        if full_key in base or full_key in ft:
            b = base.get(full_key)
            f = ft.get(full_key)
            # 両方の値が揃っているときだけ差分を計算する。
            delta = (f - b) if (b is not None and f is not None) else None
            rows.append([
                short_key,
                f"{b:.4f}" if b is not None else "—",
                f"{f:.4f}" if f is not None else "—",
                f"{delta:+.4f}" if delta is not None else "—",
            ])
    return rows


# ---------------------------------------------------------------------------
# タブ 2（データセット閲覧）用のヘルパ
# ---------------------------------------------------------------------------

def load_dataset_split(data_dir_label: str, split: str):
    """指定ディレクトリ・スプリットの datasets を読み込む（無ければ None）。"""
    from datasets import load_from_disk
    path = DATA_DIRS[data_dir_label] / split
    if not path.exists():
        return None
    return load_from_disk(str(path))


def get_sample(data_dir_label: str, split: str, idx: int):
    """指定インデックスのサンプル（画像・キャプション・カテゴリ）を取り出す。

    Returns:
        (画像, キャプション, カテゴリ, 正規化後インデックス, 総件数) のタプル。
        データが無い場合はプレースホルダを返す。
    """
    ds = load_dataset_split(data_dir_label, split)
    if ds is None or len(ds) == 0:
        return None, "データなし", "", 0, 1

    # インデックスを [0, len-1] にクランプして範囲外アクセスを防ぐ。
    idx = max(0, min(idx, len(ds) - 1))
    row = ds[idx]
    img = row["positive"]
    anchor = row["anchor"]
    category = row["category"]
    return img, anchor, category, idx, len(ds)


def dataset_nav(data_dir_label: str, split: str, idx: int, direction: str):
    """「前へ／次へ」ボタンの遷移処理。端ではラップアラウンド（循環）する。"""
    ds = load_dataset_split(data_dir_label, split)
    total = len(ds) if ds is not None else 1
    if direction == "next":
        idx = (idx + 1) % total
    elif direction == "prev":
        idx = (idx - 1) % total
    img, anchor, category, idx, total = get_sample(data_dir_label, split, idx)
    return img, anchor, category, idx, f"{idx + 1} / {total}"


# ---------------------------------------------------------------------------
# タブ 3（Reranking デモ）用のヘルパ
# ---------------------------------------------------------------------------

def load_rerank_examples(output_dir_label: str) -> list[list]:
    """rerank_examples.json を読み、表表示用の行リストへ整形する。

    リランク前後の順位を比較し、改善／変化なし／悪化のラベルを付与する。
    """
    path = OUTPUT_DIRS[output_dir_label] / "rerank_examples.json"
    if not path.exists():
        return []
    examples = json.loads(path.read_text())
    rows = []
    for ex in examples:
        rb = ex.get("rank_before_rerank")  # リランク前の正解順位
        ra = ex.get("rank_after_rerank")   # リランク後の正解順位
        improved = ""
        if rb is not None and ra is not None:
            # 順位は小さいほど上位。ra < rb なら順位が上がった＝改善。
            if ra < rb:
                improved = "↑ 改善"
            elif ra == rb:
                improved = "→ 変化なし"
            else:
                improved = "↓ 悪化"
        rows.append([
            ex.get("query", ""),
            ex.get("target", ""),
            rb if rb is not None else "—",
            ra if ra is not None else "—",
            ex.get("top_k", ""),
            improved,
        ])
    return rows


# ---------------------------------------------------------------------------
# UI 構築
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    """Gradio の Blocks アプリを組み立てて返す。"""
    output_dir_choices = list(OUTPUT_DIRS.keys())
    data_dir_choices = list(DATA_DIRS.keys())

    with gr.Blocks(title="Qwen3-VL Fine-tuning Demo Viewer", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# Qwen3-VL Fine-tuning Demo Viewer\n"
            "Qwen3-VL Embedding モデルのファインチューニング結果を可視化します。"
        )

        with gr.Tabs():

            # ----------------------------------------------------------------
            # タブ 1: メトリクス比較
            # ----------------------------------------------------------------
            with gr.Tab("📊 メトリクス比較"):
                out_dir_dd = gr.Dropdown(
                    choices=output_dir_choices,
                    value=output_dir_choices[0],
                    label="出力ディレクトリ",
                    interactive=True,
                )
                metrics_plot = gr.Plot(label="Base vs Fine-tuned メトリクス")
                metrics_table = gr.Dataframe(
                    headers=["Metric", "Base", "Fine-tuned", "Δ"],
                    datatype=["str", "str", "str", "str"],
                    label="数値比較",
                    interactive=False,
                )

                def refresh_metrics(label):
                    """ドロップダウン変更時／初期表示時にグラフと表を再生成する。"""
                    return make_metrics_figure(label), make_metrics_table(label)

                # ドロップダウン変更時と、アプリ初回ロード時の両方で更新する。
                out_dir_dd.change(refresh_metrics, inputs=out_dir_dd, outputs=[metrics_plot, metrics_table])
                demo.load(refresh_metrics, inputs=out_dir_dd, outputs=[metrics_plot, metrics_table])

            # ----------------------------------------------------------------
            # タブ 2: データセット閲覧
            # ----------------------------------------------------------------
            with gr.Tab("🖼️ データセット閲覧"):
                with gr.Row():
                    data_dir_dd = gr.Dropdown(
                        choices=data_dir_choices,
                        value=data_dir_choices[0],
                        label="データディレクトリ",
                        interactive=True,
                    )
                    split_dd = gr.Dropdown(
                        choices=["train", "eval"],
                        value="eval",
                        label="スプリット",
                        interactive=True,
                    )

                with gr.Row():
                    prev_btn = gr.Button("← 前へ", size="sm")
                    # 現在の表示インデックスを保持する非表示の状態。
                    idx_state = gr.State(value=0)
                    counter_lbl = gr.Label(label="サンプル番号", value="1 / ?")
                    next_btn = gr.Button("次へ →", size="sm")

                with gr.Row():
                    sample_img = gr.Image(label="画像", type="pil", height=350)
                    with gr.Column():
                        anchor_txt = gr.Textbox(label="キャプション (anchor)", lines=3, interactive=False)
                        category_txt = gr.Textbox(label="カテゴリ", interactive=False)

                def _load(data_dir_label, split):
                    """ディレクトリ／スプリット変更時に先頭サンプルを表示する。"""
                    img, anchor, cat, idx, total = get_sample(data_dir_label, split, 0)
                    return img, anchor, cat, 0, f"1 / {total}"

                def _prev(data_dir_label, split, idx):
                    """「前へ」ボタン。"""
                    img, anchor, cat, new_idx, counter = dataset_nav(data_dir_label, split, idx, "prev")
                    return img, anchor, cat, new_idx, counter

                def _next(data_dir_label, split, idx):
                    """「次へ」ボタン。"""
                    img, anchor, cat, new_idx, counter = dataset_nav(data_dir_label, split, idx, "next")
                    return img, anchor, cat, new_idx, counter

                # すべてのイベントが更新する出力ウィジェット群（順序が一致している必要がある）。
                _ds_outputs = [sample_img, anchor_txt, category_txt, idx_state, counter_lbl]

                data_dir_dd.change(_load, inputs=[data_dir_dd, split_dd], outputs=_ds_outputs)
                split_dd.change(_load, inputs=[data_dir_dd, split_dd], outputs=_ds_outputs)
                prev_btn.click(_prev, inputs=[data_dir_dd, split_dd, idx_state], outputs=_ds_outputs)
                next_btn.click(_next, inputs=[data_dir_dd, split_dd, idx_state], outputs=_ds_outputs)
                demo.load(_load, inputs=[data_dir_dd, split_dd], outputs=_ds_outputs)

            # ----------------------------------------------------------------
            # タブ 3: Reranking デモ
            # ----------------------------------------------------------------
            with gr.Tab("🔄 Rerankingデモ"):
                rerank_dir_dd = gr.Dropdown(
                    choices=output_dir_choices,
                    value=output_dir_choices[0],
                    label="出力ディレクトリ",
                    interactive=True,
                )
                rerank_table = gr.Dataframe(
                    headers=["クエリ", "正解画像ID", "Rerank前ランク", "Rerank後ランク", "Top-K", "結果"],
                    datatype=["str", "str", "number", "number", "number", "str"],
                    label="Rerank前後のランク比較",
                    interactive=False,
                    wrap=True,
                )

                def _load_rerank(label):
                    """リランク事例を読み込んで表に流す。データが無ければ空表を返す。"""
                    rows = load_rerank_examples(label)
                    if not rows:
                        return gr.Dataframe(value=[], headers=["クエリ", "正解画像ID", "Rerank前ランク", "Rerank後ランク", "Top-K", "結果"])
                    return rows

                rerank_dir_dd.change(_load_rerank, inputs=rerank_dir_dd, outputs=rerank_table)
                demo.load(_load_rerank, inputs=rerank_dir_dd, outputs=rerank_table)

    return demo


if __name__ == "__main__":
    # 0.0.0.0 で待受（コンテナ／リモートからアクセスできるように）。share=False で外部公開はしない。
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
