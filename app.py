"""Qwen3-VL ファインチューニング・デモの結果ビューア（Gradio）。

パイプライン（generate_data → evaluate → train → eval → rerank）が出力した
成果物を、ブラウザ上で確認するための GUI。学習を回す機能はなく、あくまで
既に生成済みの ``data*/`` と ``outputs*/`` を読んで可視化するだけの読み取り専用ツール。

タブ構成:
  1. メトリクス比較        – 埋め込みのベース vs FT 後の棒グラフ＋数値表
  2. データセット閲覧       – 生成したキャプション付き画像を 1 枚ずつブラウズ
  3. ペルソナ閲覧          – ペルソナ名→嗜好埋め込み→嗜好テキスト→生成プロンプト→生成画像の対応を可視化
  4. Reranking デモ        – rerank_examples.json からリランク前後の順位変化を表示
  5. 2 段階検索 6 パターン  – rerank_metrics.json（埋め込み{base,ft}×リランカー{base,ft,none}）を比較

起動: ``uv run python app.py`` → http://localhost:7860
"""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import japanize_matplotlib  # noqa: F401  日本語フォントを自動設定
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
    return key[len(_PREFIX) :] if key.startswith(_PREFIX) else key


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
            rows.append(
                [
                    short_key,
                    f"{b:.4f}" if b is not None else "—",
                    f"{f:.4f}" if f is not None else "—",
                    f"{delta:+.4f}" if delta is not None else "—",
                ]
            )
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
# タブ 3（ペルソナ閲覧）用のヘルパ
# ---------------------------------------------------------------------------

_PREF_MODEL_CACHE: dict[str, dict] = {}


def load_pref_model(data_dir_label: str) -> dict | None:
    """preference_model.json を読む。なければ data/ の共通ファイルへフォールバック。"""
    if data_dir_label in _PREF_MODEL_CACHE:
        return _PREF_MODEL_CACHE[data_dir_label]
    path = DATA_DIRS[data_dir_label] / "preference_model.json"
    if not path.exists():
        path = ROOT / "data" / "preference_model.json"
    if not path.exists():
        return None
    model = json.loads(path.read_text())
    _PREF_MODEL_CACHE[data_dir_label] = model
    return model


def get_persona_names(data_dir_label: str) -> list[str]:
    """選択済みデータディレクトリで利用可能なペルソナ名のリストを返す。"""
    model = load_pref_model(data_dir_label)
    if model:
        return list(model["persona_pref"].keys())
    for split in ("eval", "train"):
        ds = load_dataset_split(data_dir_label, split)
        if ds is not None:
            return sorted(set(ds["persona"]))
    return []


def make_persona_embedding_figure(data_dir_label: str, persona: str):
    """ペルソナの嗜好埋め込みを7軸の横棒グラフで描く。"""
    model = load_pref_model(data_dir_label)
    if model is None or persona not in model.get("persona_pref", {}):
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "preference_model.json が見つかりません", ha="center", va="center")
        return fig

    axes_labels = model["axes"]
    vec = model["persona_pref"][persona]
    colors = ["#DD8452" if v > 0 else "#4C72B0" if v < 0 else "#aaaaaa" for v in vec]

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.barh(axes_labels, vec, color=colors, alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlim(-1.3, 1.3)
    ax.set_xlabel("嗜好強度（+ 好む / − 嫌う）", fontsize=9)
    ax.set_title(f"{persona} の嗜好埋め込み", fontsize=10, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


def get_persona_pref_text(data_dir_label: str, persona: str) -> str:
    """嗜好埋め込みをこだわりの強い軸順にテキストで返す。"""
    model = load_pref_model(data_dir_label)
    if model is None or persona not in model.get("persona_pref", {}):
        return "データなし"
    axes_labels = model["axes"]
    vec = model["persona_pref"][persona]
    fragments = model["fragments"]
    ranked = sorted(range(len(vec)), key=lambda i: abs(vec[i]), reverse=True)
    lines = []
    for i in ranked:
        v = vec[i]
        if abs(v) < 1e-9:
            continue
        frag = fragments[axes_labels[i]][1 if v > 0 else 0]
        lines.append(f"{axes_labels[i]:12s}  {v:+.2f}  →  {frag}")
    return "\n".join(lines) if lines else "嗜好ベクトルがゼロです"


def get_persona_archetype_text(data_dir_label: str, persona: str) -> str:
    """アーキタイプ混合比を文字列で返す。"""
    model = load_pref_model(data_dir_label)
    if model is None or persona not in model.get("persona_mix", {}):
        return "データなし"
    mix = model["persona_mix"][persona]
    return "  +  ".join(f"{arch} × {w:.1f}" for arch, w in mix.items())


def make_archetype_table(data_dir_label: str) -> tuple[list[str], list[list]]:
    """全アーキタイプ × 7 軸の定義テーブル（ヘッダ, 行）を返す。ペルソナ非依存。"""
    model = load_pref_model(data_dir_label)
    if model is None:
        return ["アーキタイプ"], []
    axes = model["axes"]
    headers = ["アーキタイプ"] + axes
    rows = [[arch] + [f"{v:+.1f}" for v in vec] for arch, vec in model["archetypes"].items()]
    return headers, rows


def get_persona_calc_text(data_dir_label: str, persona: str) -> str:
    """嗜好埋め込みの計算式（アーキタイプ加重和）を展開したテキストを返す。"""
    model = load_pref_model(data_dir_label)
    if model is None or persona not in model.get("persona_pref", {}):
        return "データなし"
    axes = model["axes"]
    mix = model["persona_mix"].get(persona, {})
    archetypes = model["archetypes"]
    vec = model["persona_pref"][persona]

    mix_str = "  +  ".join(f"{w:.1f}×{arch}" for arch, w in mix.items())
    lines = [f"[計算式]  persona_pref = {mix_str}", ""]
    for arch, w in mix.items():
        av = archetypes.get(arch, [0.0] * len(axes))
        av_str = "[" + ", ".join(f"{v:+.1f}" for v in av) + "]"
        lines.append(f"  {arch:16s} = {av_str}  ×{w:.1f}")
    lines.append("  " + "─" * 58)
    result_str = "[" + ", ".join(f"{v:+.2f}" for v in vec) + "]"
    lines.append(f"  {'persona_pref':16s} = {result_str}")
    lines.append("")
    lines.append("  軸順序: " + ", ".join(axes))
    return "\n".join(lines)


def get_persona_interaction_text(data_dir_label: str, persona: str) -> str:
    """このペルソナの非加法的交互作用（INTERACTIONS）を人間可読テキストで返す。"""
    model = load_pref_model(data_dir_label)
    if model is None:
        return "データなし"
    interactions = model.get("interactions", {}).get(persona, [])
    if not interactions:
        return "（交互作用なし）"
    axes = model["axes"]
    gamma = model.get("gamma", 2.0)
    lines = [f"[非加法的交互作用]  appeal への加算項: γ={gamma:.1f} × coef × (a_i AND a_j)", ""]
    for tri in interactions:
        i, j, coef = int(tri[0]), int(tri[1]), tri[2]
        sign = "+" if coef >= 0 else ""
        note = "加点（両方1のとき好む）" if coef > 0 else "減点（両方1のとき嫌う）"
        lines.append(f"  {axes[i]:12s} ∧ {axes[j]:12s}  →  {sign}{coef:.1f}   {note}")
    return "\n".join(lines)


def _persona_rows(data_dir_label: str, persona: str) -> list[dict]:
    """eval → train の順で探し、ペルソナ一致の行リストを返す。"""
    for split in ("eval", "train"):
        ds = load_dataset_split(data_dir_label, split)
        if ds is None:
            continue
        rows = [ds[i] for i in range(len(ds)) if ds[i]["persona"] == persona]
        if rows:
            return rows
    return []


def get_persona_image(data_dir_label: str, persona: str, idx: int):
    """ペルソナの idx 番目サンプルを返す。"""
    rows = _persona_rows(data_dir_label, persona)
    if not rows:
        return None, "（このペルソナのデータなし）", "", "", 0, 0
    idx = max(0, min(idx, len(rows) - 1))
    row = rows[idx]
    return row["positive"], row["anchor"], row["subject"], row["category"], idx, len(rows)


def persona_image_nav(data_dir_label: str, persona: str, idx: int, direction: str):
    """前へ／次へ遷移。端でラップアラウンド。"""
    rows = _persona_rows(data_dir_label, persona)
    total = len(rows)
    if total == 0:
        return None, "（このペルソナのデータなし）", "", "", 0, "0 / 0"
    if direction == "next":
        idx = (idx + 1) % total
    elif direction == "prev":
        idx = (idx - 1) % total
    img, anchor, subject, category, idx, total = get_persona_image(data_dir_label, persona, idx)
    return img, anchor, subject, category, idx, f"{idx + 1} / {total}"


# ---------------------------------------------------------------------------
# タブ 4（Reranking デモ）用のヘルパ
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
        rb = ex.get("best_rank_before_rerank")  # リランク前の最良正解順位
        ra = ex.get("best_rank_after_rerank")  # リランク後の最良正解順位
        improved = ""
        if rb is not None and ra is not None:
            # 順位は小さいほど上位。ra < rb なら順位が上がった＝改善。
            if ra < rb:
                improved = "↑ 改善"
            elif ra == rb:
                improved = "→ 変化なし"
            else:
                improved = "↓ 悪化"
        rows.append(
            [
                ex.get("query", ""),
                ex.get("num_relevant", ""),
                rb if rb is not None else "—",
                ra if ra is not None else "—",
                ex.get("top_k", ""),
                improved,
            ]
        )
    return rows


# ---------------------------------------------------------------------------
# タブ 4（2 段階検索の 6 パターン評価）用のヘルパ
# ---------------------------------------------------------------------------

# 2 段階パターン（埋め込み+リランカー）の表示順。先頭 4 つが主要 4 パターン、
# 末尾の rerank=none は「リランクなし（埋め込み検索のみ）」の参考値。
_PATTERN_ORDER = [
    "embed=base+rerank=base",
    "embed=ft+rerank=base",
    "embed=base+rerank=ft",
    "embed=ft+rerank=ft",
    "embed=base+rerank=none",
    "embed=ft+rerank=none",
]


def _pattern_label(key: str) -> str:
    """'embed=ft+rerank=base' -> 'ft+base' のように短い表示名にする。"""
    return key.replace("embed=", "").replace("rerank=", "")


def _metric_sort_key(metric: str):
    """メトリクスキーを ndcg → recall → accuracy → mrr → map、各 @k 昇順で並べる。"""
    priority = {"ndcg": 0, "recall": 1, "accuracy": 2, "mrr": 3, "map": 4}
    name, _, k = metric.partition("@")
    try:
        kk = int(k) if k else -1
    except ValueError:
        kk = -1
    return (priority.get(name, 9), kk)


def load_rerank_metrics(output_dir_label: str) -> dict:
    """rerank_metrics.json（6 パターンの検索指標）を読み込む。無ければ空 dict。"""
    path = OUTPUT_DIRS[output_dir_label] / "rerank_metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _ordered_patterns(metrics: dict) -> list[str]:
    """既知の表示順を優先しつつ、未知のキーは末尾に回す。"""
    ordered = [k for k in _PATTERN_ORDER if k in metrics]
    ordered += [k for k in metrics if k not in ordered]
    return ordered


def _ordered_metric_keys(metrics: dict) -> list[str]:
    """全パターンに現れるメトリクスキーの和集合を、見やすい順に並べる。"""
    keys: set[str] = set()
    for v in metrics.values():
        keys.update(v.keys())
    return sorted(keys, key=_metric_sort_key)


def make_rerank_metrics_table(output_dir_label: str) -> tuple[list[str], list[list]]:
    """6 パターン×全メトリクスの表（ヘッダ, 行）を返す。"""
    metrics = load_rerank_metrics(output_dir_label)
    if not metrics:
        return ["パターン"], []
    mkeys = _ordered_metric_keys(metrics)
    headers = ["パターン (埋め込み+リランカー)"] + mkeys
    rows = []
    for key in _ordered_patterns(metrics):
        v = metrics[key]
        rows.append([_pattern_label(key)] + [f"{v.get(m, 0.0):.4f}" for m in mkeys])
    return headers, rows


def make_rerank_metrics_figure(output_dir_label: str):
    """主要 4 パターンを各メトリクスでグループ化した棒グラフを作る。"""
    metrics = load_rerank_metrics(output_dir_label)
    if not metrics:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "rerank_metrics.json が見つかりません", ha="center", va="center")
        return fig

    # 図は主要 4 パターンに絞る（rerank=none の参考値は表のみ）。
    patterns = [k for k in _ordered_patterns(metrics) if not k.endswith("rerank=none")]
    if not patterns:
        patterns = _ordered_patterns(metrics)
    mkeys = _ordered_metric_keys(metrics)

    x = np.arange(len(mkeys))
    width = 0.8 / max(1, len(patterns))

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, key in enumerate(patterns):
        vals = [metrics[key].get(m, 0.0) for m in mkeys]
        offset = (i - (len(patterns) - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=_pattern_label(key))

    ax.set_xticks(x)
    ax.set_xticklabels(mkeys, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(
        f"2 段階検索 主要 4 パターン比較 — {output_dir_label}", fontsize=11, fontweight="bold"
    )
    ax.legend(title="埋め込み+リランカー", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


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
                out_dir_dd.change(
                    refresh_metrics, inputs=out_dir_dd, outputs=[metrics_plot, metrics_table]
                )
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
                        anchor_txt = gr.Textbox(
                            label="キャプション (anchor)", lines=3, interactive=False
                        )
                        category_txt = gr.Textbox(label="カテゴリ", interactive=False)

                def _load(data_dir_label, split):
                    """ディレクトリ／スプリット変更時に先頭サンプルを表示する。"""
                    img, anchor, cat, idx, total = get_sample(data_dir_label, split, 0)
                    return img, anchor, cat, 0, f"1 / {total}"

                def _prev(data_dir_label, split, idx):
                    """「前へ」ボタン。"""
                    img, anchor, cat, new_idx, counter = dataset_nav(
                        data_dir_label, split, idx, "prev"
                    )
                    return img, anchor, cat, new_idx, counter

                def _next(data_dir_label, split, idx):
                    """「次へ」ボタン。"""
                    img, anchor, cat, new_idx, counter = dataset_nav(
                        data_dir_label, split, idx, "next"
                    )
                    return img, anchor, cat, new_idx, counter

                # すべてのイベントが更新する出力ウィジェット群（順序が一致している必要がある）。
                _ds_outputs = [sample_img, anchor_txt, category_txt, idx_state, counter_lbl]

                data_dir_dd.change(_load, inputs=[data_dir_dd, split_dd], outputs=_ds_outputs)
                split_dd.change(_load, inputs=[data_dir_dd, split_dd], outputs=_ds_outputs)
                prev_btn.click(
                    _prev, inputs=[data_dir_dd, split_dd, idx_state], outputs=_ds_outputs
                )
                next_btn.click(
                    _next, inputs=[data_dir_dd, split_dd, idx_state], outputs=_ds_outputs
                )
                demo.load(_load, inputs=[data_dir_dd, split_dd], outputs=_ds_outputs)

            # ----------------------------------------------------------------
            # タブ 3: ペルソナ閲覧
            # ----------------------------------------------------------------
            with gr.Tab("🧑 ペルソナ閲覧"):
                gr.Markdown(
                    "**所与 → 計算 → 変換** の流れでペルソナの嗜好構造を確認します。  \n"
                    "左列: アーキタイプ混合（所与）→ 加重和（計算）→ 嗜好埋め込み（結果）"
                    "→ 嗜好テキスト（FRAGMENTS 変換）→ 交互作用（非線形補正）  \n"
                    "右列: そのペルソナの生成画像と生成用プロンプトをブラウズ"
                )
                with gr.Row():
                    p_data_dir_dd = gr.Dropdown(
                        choices=data_dir_choices,
                        value=data_dir_choices[0],
                        label="データディレクトリ",
                        interactive=True,
                    )
                    p_persona_dd = gr.Dropdown(
                        choices=get_persona_names(data_dir_choices[0]),
                        value=(get_persona_names(data_dir_choices[0]) or [""])[0],
                        label="ペルソナ",
                        interactive=True,
                    )

                # アーキタイプ定義テーブル（ペルソナ非依存・折りたたみ）
                with gr.Accordion(
                    "📐 所与: アーキタイプ定義（嗜好空間の基底ベクトル）", open=False
                ):
                    gr.Markdown(
                        "6 種類のアーキタイプが嗜好空間の「型」を定義します。"
                        "各値は **+1**（好む）/ **−1**（嫌う）/ **0**（無関心）。"
                        "軸順序: warmth / era / ornament / mood / saturation / material / setting"
                    )
                    _init_arch_headers, _init_arch_rows = make_archetype_table(data_dir_choices[0])
                    p_arch_table = gr.Dataframe(
                        value=_init_arch_rows,
                        headers=_init_arch_headers,
                        label="アーキタイプ × 軸",
                        interactive=False,
                    )

                with gr.Row():
                    # 左: 計算過程
                    with gr.Column(scale=1):
                        p_archetype_txt = gr.Textbox(
                            label="所与: アーキタイプ混合（凸結合の重み）",
                            interactive=False,
                            lines=2,
                        )
                        p_calc_txt = gr.Textbox(
                            label="計算: 加重和の展開式  →  persona_pref",
                            interactive=False,
                            lines=7,
                        )
                        p_embed_plot = gr.Plot(label="結果: 嗜好埋め込み（7軸）")
                        p_pref_txt = gr.Textbox(
                            label="変換: 嗜好テキスト（FRAGMENTS 経由、こだわり強度順）",
                            interactive=False,
                            lines=8,
                        )
                        p_interaction_txt = gr.Textbox(
                            label="補正: 非加法的交互作用（INTERACTIONS）",
                            interactive=False,
                            lines=5,
                        )

                    # 右: 画像ブラウザ
                    with gr.Column(scale=1):
                        with gr.Row():
                            p_prev_btn = gr.Button("← 前へ", size="sm")
                            p_idx_state = gr.State(value=0)
                            p_counter_lbl = gr.Label(label="サンプル番号", value="— / —")
                            p_next_btn = gr.Button("次へ →", size="sm")
                        p_img = gr.Image(label="生成画像", type="pil", height=380)
                        p_anchor_txt = gr.Textbox(
                            label="生成用プロンプト（anchor）", interactive=False, lines=4
                        )
                        with gr.Row():
                            p_subject_txt = gr.Textbox(label="被写体", interactive=False)
                            p_category_txt = gr.Textbox(label="カテゴリ", interactive=False)

                # 左ペインの出力リスト（アーキタイプテーブルはペルソナ非依存なので別管理）
                _p_left = [p_archetype_txt, p_calc_txt, p_embed_plot, p_pref_txt, p_interaction_txt]
                # 右ペインの出力リスト
                _p_right = [
                    p_img,
                    p_anchor_txt,
                    p_subject_txt,
                    p_category_txt,
                    p_idx_state,
                    p_counter_lbl,
                ]

                def _p_on_data_dir(data_dir_label):
                    """データディレクトリ変更時: ペルソナ選択肢・アーキタイプテーブル・全パネルを更新。"""
                    names = get_persona_names(data_dir_label)
                    persona = names[0] if names else ""
                    headers, rows = make_archetype_table(data_dir_label)
                    arch = get_persona_archetype_text(data_dir_label, persona)
                    calc = get_persona_calc_text(data_dir_label, persona)
                    fig = make_persona_embedding_figure(data_dir_label, persona)
                    pref = get_persona_pref_text(data_dir_label, persona)
                    inter = get_persona_interaction_text(data_dir_label, persona)
                    img, anchor, subject, cat, idx, total = get_persona_image(
                        data_dir_label, persona, 0
                    )
                    counter = f"1 / {total}" if total > 0 else "0 / 0"
                    return (
                        gr.update(choices=names, value=persona),
                        gr.Dataframe(value=rows, headers=headers),
                        arch,
                        calc,
                        fig,
                        pref,
                        inter,
                        img,
                        anchor,
                        subject,
                        cat,
                        0,
                        counter,
                    )

                def _p_on_persona(data_dir_label, persona):
                    arch = get_persona_archetype_text(data_dir_label, persona)
                    calc = get_persona_calc_text(data_dir_label, persona)
                    fig = make_persona_embedding_figure(data_dir_label, persona)
                    pref = get_persona_pref_text(data_dir_label, persona)
                    inter = get_persona_interaction_text(data_dir_label, persona)
                    img, anchor, subject, cat, idx, total = get_persona_image(
                        data_dir_label, persona, 0
                    )
                    counter = f"1 / {total}" if total > 0 else "0 / 0"
                    return arch, calc, fig, pref, inter, img, anchor, subject, cat, 0, counter

                def _p_prev(data_dir_label, persona, idx):
                    img, anchor, subject, cat, new_idx, counter = persona_image_nav(
                        data_dir_label, persona, idx, "prev"
                    )
                    return img, anchor, subject, cat, new_idx, counter

                def _p_next(data_dir_label, persona, idx):
                    img, anchor, subject, cat, new_idx, counter = persona_image_nav(
                        data_dir_label, persona, idx, "next"
                    )
                    return img, anchor, subject, cat, new_idx, counter

                _p_all_outputs = [p_persona_dd, p_arch_table] + _p_left + _p_right
                _p_persona_outputs = _p_left + _p_right

                p_data_dir_dd.change(_p_on_data_dir, inputs=p_data_dir_dd, outputs=_p_all_outputs)
                p_persona_dd.change(
                    _p_on_persona,
                    inputs=[p_data_dir_dd, p_persona_dd],
                    outputs=_p_persona_outputs,
                )
                p_prev_btn.click(
                    _p_prev,
                    inputs=[p_data_dir_dd, p_persona_dd, p_idx_state],
                    outputs=_p_right,
                )
                p_next_btn.click(
                    _p_next,
                    inputs=[p_data_dir_dd, p_persona_dd, p_idx_state],
                    outputs=_p_right,
                )
                demo.load(
                    _p_on_persona,
                    inputs=[p_data_dir_dd, p_persona_dd],
                    outputs=_p_persona_outputs,
                )

            # ----------------------------------------------------------------
            # タブ 4: Reranking デモ
            # ----------------------------------------------------------------
            with gr.Tab("🔄 Rerankingデモ"):
                rerank_dir_dd = gr.Dropdown(
                    choices=output_dir_choices,
                    value=output_dir_choices[0],
                    label="出力ディレクトリ",
                    interactive=True,
                )
                rerank_table = gr.Dataframe(
                    headers=[
                        "クエリ",
                        "関連画像数",
                        "Rerank前ランク",
                        "Rerank後ランク",
                        "Top-K",
                        "結果",
                    ],
                    datatype=["str", "number", "number", "number", "number", "str"],
                    label="Rerank前後のランク比較",
                    interactive=False,
                    wrap=True,
                )

                def _load_rerank(label):
                    """リランク事例を読み込んで表に流す。データが無ければ空表を返す。"""
                    rows = load_rerank_examples(label)
                    if not rows:
                        return gr.Dataframe(
                            value=[],
                            headers=[
                                "クエリ",
                                "関連画像数",
                                "Rerank前ランク",
                                "Rerank後ランク",
                                "Top-K",
                                "結果",
                            ],
                        )
                    return rows

                rerank_dir_dd.change(_load_rerank, inputs=rerank_dir_dd, outputs=rerank_table)
                demo.load(_load_rerank, inputs=rerank_dir_dd, outputs=rerank_table)

            # ----------------------------------------------------------------
            # タブ 5: 2 段階検索 6 パターン評価
            # ----------------------------------------------------------------
            with gr.Tab("🔀 2段階検索 (6パターン)"):
                gr.Markdown(
                    "埋め込み{base, ft} × リランカー{base, ft, none} の **6 パターン**で "
                    "2 段階検索（retrieve → rerank）の精度を比較します"
                    "（`rerank_metrics.json`）。`rerank=none` は埋め込み検索のみの参考値です。"
                    "（グラフは主要 4 パターン、表は 6 パターンすべてを表示します。）"
                )
                rr4_dir_dd = gr.Dropdown(
                    choices=output_dir_choices,
                    value=output_dir_choices[0],
                    label="出力ディレクトリ",
                    interactive=True,
                )
                rr4_plot = gr.Plot(label="主要 4 パターン比較（埋め込み × リランカー）")
                rr4_table = gr.Dataframe(label="全メトリクス", interactive=False, wrap=True)

                def _refresh_rr4(label):
                    """ドロップダウン変更時／初期表示時にグラフと表を再生成する。"""
                    headers, rows = make_rerank_metrics_table(label)
                    return make_rerank_metrics_figure(label), gr.Dataframe(
                        value=rows, headers=headers
                    )

                rr4_dir_dd.change(_refresh_rr4, inputs=rr4_dir_dd, outputs=[rr4_plot, rr4_table])
                demo.load(_refresh_rr4, inputs=rr4_dir_dd, outputs=[rr4_plot, rr4_table])

    return demo


if __name__ == "__main__":
    # 0.0.0.0 で待受（コンテナ／リモートからアクセスできるように）。share=False で外部公開はしない。
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
