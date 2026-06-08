"""Gradio viewer for the Qwen3-VL fine-tuning demo.

Tabs:
  1. メトリクス比較  – bar chart: base vs finetuned for any output dir
  2. データセット閲覧 – browse captioned images from any data dir
  3. Rerankingデモ   – before/after rerank table from rerank_examples.json
"""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent

OUTPUT_DIRS = {
    "outputs (full run)": ROOT / "outputs",
    "outputs_smoke (smoke run)": ROOT / "outputs_smoke",
}

DATA_DIRS = {
    "data (full run)": ROOT / "data",
    "data_smoke (smoke run)": ROOT / "data_smoke",
}

# Metrics we care about, in display order
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

_PREFIX = "synthetic-image-retrieval_cosine_"


def _strip_prefix(key: str) -> str:
    return key[len(_PREFIX):] if key.startswith(_PREFIX) else key


# ---------------------------------------------------------------------------
# Tab 1 helpers
# ---------------------------------------------------------------------------

def load_metrics(output_dir_label: str) -> tuple[dict, dict]:
    out_dir = OUTPUT_DIRS[output_dir_label]
    base_path = out_dir / "metrics_base.json"
    ft_path = out_dir / "metrics_finetuned.json"

    base = json.loads(base_path.read_text()) if base_path.exists() else {}
    ft = json.loads(ft_path.read_text()) if ft_path.exists() else {}
    return base, ft


def make_metrics_figure(output_dir_label: str):
    base, ft = load_metrics(output_dir_label)

    labels, base_vals, ft_vals = [], [], []
    for short_key in KEY_METRICS:
        full_key = _PREFIX + short_key
        if full_key in base or full_key in ft:
            labels.append(short_key)
            base_vals.append(base.get(full_key, 0.0))
            ft_vals.append(ft.get(full_key, 0.0))

    if not labels:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "メトリクスデータが見つかりません", ha="center", va="center")
        return fig

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    bars_base = ax.bar(x - width / 2, base_vals, width, label="Base", color="#4C72B0", alpha=0.85)
    bars_ft = ax.bar(x + width / 2, ft_vals, width, label="Fine-tuned", color="#DD8452", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title(f"Base vs Fine-tuned — {output_dir_label}", fontsize=11, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    def _annotate(bars):
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
    base, ft = load_metrics(output_dir_label)
    rows = []
    for short_key in KEY_METRICS:
        full_key = _PREFIX + short_key
        if full_key in base or full_key in ft:
            b = base.get(full_key)
            f = ft.get(full_key)
            delta = (f - b) if (b is not None and f is not None) else None
            rows.append([
                short_key,
                f"{b:.4f}" if b is not None else "—",
                f"{f:.4f}" if f is not None else "—",
                f"{delta:+.4f}" if delta is not None else "—",
            ])
    return rows


# ---------------------------------------------------------------------------
# Tab 2 helpers
# ---------------------------------------------------------------------------

def load_dataset_split(data_dir_label: str, split: str):
    from datasets import load_from_disk
    path = DATA_DIRS[data_dir_label] / split
    if not path.exists():
        return None
    return load_from_disk(str(path))


def get_sample(data_dir_label: str, split: str, idx: int):
    ds = load_dataset_split(data_dir_label, split)
    if ds is None or len(ds) == 0:
        return None, "データなし", "", 0, 1

    idx = max(0, min(idx, len(ds) - 1))
    row = ds[idx]
    img = row["positive"]
    anchor = row["anchor"]
    category = row["category"]
    return img, anchor, category, idx, len(ds)


def dataset_nav(data_dir_label: str, split: str, idx: int, direction: str):
    ds = load_dataset_split(data_dir_label, split)
    total = len(ds) if ds is not None else 1
    if direction == "next":
        idx = (idx + 1) % total
    elif direction == "prev":
        idx = (idx - 1) % total
    img, anchor, category, idx, total = get_sample(data_dir_label, split, idx)
    return img, anchor, category, idx, f"{idx + 1} / {total}"


# ---------------------------------------------------------------------------
# Tab 3 helpers
# ---------------------------------------------------------------------------

def load_rerank_examples(output_dir_label: str) -> list[list]:
    path = OUTPUT_DIRS[output_dir_label] / "rerank_examples.json"
    if not path.exists():
        return []
    examples = json.loads(path.read_text())
    rows = []
    for ex in examples:
        rb = ex.get("rank_before_rerank")
        ra = ex.get("rank_after_rerank")
        improved = ""
        if rb is not None and ra is not None:
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
# Build UI
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    output_dir_choices = list(OUTPUT_DIRS.keys())
    data_dir_choices = list(DATA_DIRS.keys())

    with gr.Blocks(title="Qwen3-VL Fine-tuning Demo Viewer", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# Qwen3-VL Fine-tuning Demo Viewer\n"
            "Qwen3-VL Embedding モデルのファインチューニング結果を可視化します。"
        )

        with gr.Tabs():

            # ----------------------------------------------------------------
            # Tab 1: メトリクス比較
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
                    return make_metrics_figure(label), make_metrics_table(label)

                out_dir_dd.change(refresh_metrics, inputs=out_dir_dd, outputs=[metrics_plot, metrics_table])
                demo.load(refresh_metrics, inputs=out_dir_dd, outputs=[metrics_plot, metrics_table])

            # ----------------------------------------------------------------
            # Tab 2: データセット閲覧
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
                    idx_state = gr.State(value=0)
                    counter_lbl = gr.Label(label="サンプル番号", value="1 / ?")
                    next_btn = gr.Button("次へ →", size="sm")

                with gr.Row():
                    sample_img = gr.Image(label="画像", type="pil", height=350)
                    with gr.Column():
                        anchor_txt = gr.Textbox(label="キャプション (anchor)", lines=3, interactive=False)
                        category_txt = gr.Textbox(label="カテゴリ", interactive=False)

                def _load(data_dir_label, split):
                    img, anchor, cat, idx, total = get_sample(data_dir_label, split, 0)
                    return img, anchor, cat, 0, f"1 / {total}"

                def _prev(data_dir_label, split, idx):
                    img, anchor, cat, new_idx, counter = dataset_nav(data_dir_label, split, idx, "prev")
                    return img, anchor, cat, new_idx, counter

                def _next(data_dir_label, split, idx):
                    img, anchor, cat, new_idx, counter = dataset_nav(data_dir_label, split, idx, "next")
                    return img, anchor, cat, new_idx, counter

                _ds_outputs = [sample_img, anchor_txt, category_txt, idx_state, counter_lbl]

                data_dir_dd.change(_load, inputs=[data_dir_dd, split_dd], outputs=_ds_outputs)
                split_dd.change(_load, inputs=[data_dir_dd, split_dd], outputs=_ds_outputs)
                prev_btn.click(_prev, inputs=[data_dir_dd, split_dd, idx_state], outputs=_ds_outputs)
                next_btn.click(_next, inputs=[data_dir_dd, split_dd, idx_state], outputs=_ds_outputs)
                demo.load(_load, inputs=[data_dir_dd, split_dd], outputs=_ds_outputs)

            # ----------------------------------------------------------------
            # Tab 3: Rerankingデモ
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
                    rows = load_rerank_examples(label)
                    if not rows:
                        return gr.Dataframe(value=[], headers=["クエリ", "正解画像ID", "Rerank前ランク", "Rerank後ランク", "Top-K", "結果"])
                    return rows

                rerank_dir_dd.change(_load_rerank, inputs=rerank_dir_dd, outputs=rerank_table)
                demo.load(_load_rerank, inputs=rerank_dir_dd, outputs=rerank_table)

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
