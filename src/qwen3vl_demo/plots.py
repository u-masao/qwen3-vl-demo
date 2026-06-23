"""チャート描画の純コア（app.py と figures.py の単一ソース）。

Gradio 画面（``app.py``）と PNG 書き出し（``figures.py``）の双方が、ここにある描画関数を
共有して使う。各 ``plot_*`` 関数は「すでにロード済みのデータ（dict / list / 数値）」と
matplotlib だけを受け取り Figure を返す純関数で、ファイル読込・パス解決・モデル計算は
呼び出し側に委ねる（app は ``OUTPUT_DIRS`` 系、figures は ``cfg`` / 嗜好モデル）。こうして
「描画ロジックは 1 か所」を保ちつつ、画面表示にも PNG 出力にも同じ図を使えるようにする。

matplotlib は :func:`_setup_plt` で遅延 import する（このモジュールを import しただけでは
重い描画依存を読み込まない。``figures.py`` の従来方針を踏襲）。numpy は datasets / torch が
必ず引く軽い依存なのでトップレベルで import する。
"""

from __future__ import annotations

import textwrap

import numpy as np

# --- 配色 -------------------------------------------------------------------
# Base / 負 / 0 は寒色、Fine-tuned / 正は暖色、正解ヒットは緑、で統一する（app.py 由来）。
COLOR_BASE = "#4C72B0"
COLOR_FT = "#DD8452"
COLOR_ZERO = "#aaaaaa"
COLOR_HIT = "#2ca02c"

# --- メトリクスキーの規約（evaluate.py / app.py と対応）--------------------
# 評価器が付ける接頭辞。例: "synthetic-image-retrieval_cosine_ndcg@10" → "ndcg@10"。
METRIC_PREFIX = "synthetic-image-retrieval_cosine_"

# メトリクス比較で表示する主要キー（表示順）。
DEFAULT_KEY_METRICS = [
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

# 2 段階検索パターンの表示順（先頭 4 つが主要 4 パターン、末尾は rerank=none の参考値）。
PATTERN_ORDER = [
    "embed=base+rerank=base",
    "embed=ft+rerank=base",
    "embed=base+rerank=ft",
    "embed=ft+rerank=ft",
    "embed=base+rerank=none",
    "embed=ft+rerank=none",
]


def _setup_plt():
    """非表示（Agg）バックエンドで matplotlib を用意して plt を返す。

    matplotlib / japanize-matplotlib はここで遅延 import する。図を作らない経路
    （他モジュールからの import 等）で重い描画依存を読み込まないため。
    """
    import matplotlib

    matplotlib.use("Agg")  # ディスプレイの無い環境でも PNG を書き出せるように
    import matplotlib.pyplot as plt

    try:  # 日本語の軸ラベル・タイトルが文字化けしないように（任意依存扱い）
        import japanize_matplotlib  # noqa: F401
    except Exception:  # noqa: BLE001 - 無くても英語ラベルなら問題ない
        pass
    return plt


def placeholder_figure(message: str):
    """データ欠如時などに「○○が見つかりません」と中央に書いた Figure を返す。

    app の各タブと figures の build_* が、前提データが無いときの共通フォールバックに使う。
    """
    plt = _setup_plt()
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12, wrap=True)
    ax.axis("off")
    return fig


# ---------------------------------------------------------------------------
# 純ロジックヘルパ（メトリクスキーの整形・並べ替え）。app.py / figures.py で共有。
# ---------------------------------------------------------------------------


def strip_prefix(key: str) -> str:
    """メトリクスキーから接頭辞を除いて短い表示名にする。"""
    return key[len(METRIC_PREFIX) :] if key.startswith(METRIC_PREFIX) else key


def pattern_label(key: str) -> str:
    """'embed=ft+rerank=base' -> 'ft+base' のように短い表示名にする。"""
    return key.replace("embed=", "").replace("rerank=", "")


def metric_sort_key(metric: str):
    """メトリクスキーを ndcg → recall → accuracy → mrr → map、各 @k 昇順で並べる。"""
    priority = {"ndcg": 0, "recall": 1, "accuracy": 2, "mrr": 3, "map": 4}
    name, _, k = metric.partition("@")
    try:
        kk = int(k) if k else -1
    except ValueError:
        kk = -1
    return (priority.get(name, 9), kk)


def ordered_patterns(metrics: dict) -> list[str]:
    """既知の表示順を優先しつつ、未知のキーは末尾に回す。"""
    ordered = [k for k in PATTERN_ORDER if k in metrics]
    ordered += [k for k in metrics if k not in ordered]
    return ordered


def ordered_metric_keys(metrics: dict) -> list[str]:
    """全パターンに現れるメトリクスキーの和集合を、見やすい順に並べる。"""
    keys: set[str] = set()
    for v in metrics.values():
        keys.update(v.keys())
    return sorted(keys, key=metric_sort_key)


def bar_colors(vec) -> list[str]:
    """嗜好強度ベクトルの符号に応じた配色（正→暖色 / 負→寒色 / 0→灰）を返す。"""
    return [COLOR_FT if v > 0 else COLOR_BASE if v < 0 else COLOR_ZERO for v in vec]


# ---------------------------------------------------------------------------
# 既存 3 チャート（app.py のタブ 1 / 3 / 5 と共有）
# ---------------------------------------------------------------------------


def plot_metrics(base: dict, ft: dict, title: str, key_metrics: list[str] | None = None):
    """ベース vs ファインチューニング後を並べた棒グラフ（matplotlib Figure）を作る。"""
    if key_metrics is None:
        key_metrics = DEFAULT_KEY_METRICS

    # KEY_METRICS のうち、どちらかのファイルに存在する項目だけを採用する。
    labels, base_vals, ft_vals = [], [], []
    for short_key in key_metrics:
        full_key = METRIC_PREFIX + short_key
        if full_key in base or full_key in ft:
            labels.append(short_key)
            base_vals.append(base.get(full_key, 0.0))
            ft_vals.append(ft.get(full_key, 0.0))

    if not labels:
        return placeholder_figure("メトリクスデータが見つかりません")

    plt = _setup_plt()
    x = np.arange(len(labels))
    width = 0.35  # 棒の幅（ベースと FT を左右にずらして並べる）

    fig, ax = plt.subplots(figsize=(12, 5))
    bars_base = ax.bar(x - width / 2, base_vals, width, label="Base", color=COLOR_BASE, alpha=0.85)
    bars_ft = ax.bar(x + width / 2, ft_vals, width, label="Fine-tuned", color=COLOR_FT, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 1.12)  # スコアは 0〜1。注釈ラベル用に上を少し余らせる。
    ax.set_ylabel("Score")
    ax.set_title(title, fontsize=11, fontweight="bold")
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


def plot_rerank_metrics(metrics: dict, title: str):
    """主要 4 パターン（埋め込み{base,ft}×リランカー{base,ft}）を各メトリクスで比較する棒グラフ。"""
    if not metrics:
        return placeholder_figure("rerank_metrics.json が見つかりません")

    # 図は主要 4 パターンに絞る（rerank=none の参考値は表のみ）。
    patterns = [k for k in ordered_patterns(metrics) if not k.endswith("rerank=none")]
    if not patterns:
        patterns = ordered_patterns(metrics)
    mkeys = ordered_metric_keys(metrics)

    plt = _setup_plt()
    x = np.arange(len(mkeys))
    width = 0.8 / max(1, len(patterns))

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, key in enumerate(patterns):
        vals = [metrics[key].get(m, 0.0) for m in mkeys]
        offset = (i - (len(patterns) - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=pattern_label(key))

    ax.set_xticks(x)
    ax.set_xticklabels(mkeys, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(title="埋め込み+リランカー", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


def plot_persona_embedding(axes_labels: list[str], vec: list[float], persona: str):
    """ペルソナの嗜好埋め込みを 7 軸の横棒グラフで描く。"""
    plt = _setup_plt()
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.barh(axes_labels, vec, color=bar_colors(vec), alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlim(-1.3, 1.3)
    ax.set_xlabel("嗜好強度（+ 好む / − 嫌う）", fontsize=9)
    ax.set_title(f"{persona} の嗜好埋め込み", fontsize=10, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 嗜好モデルの構造を説明する図（preference_model.json の静的データのみで描ける）
# ---------------------------------------------------------------------------


def _heatmap(ax, matrix, row_labels, col_labels, *, vmin, vmax, cmap, fmt):
    """共通のヒートマップ描画（セルに数値注釈つき）。``imshow`` のハンドルを返す。"""
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    span = (vmax - vmin) or 1.0
    for r in range(len(row_labels)):
        for c in range(len(col_labels)):
            v = matrix[r][c]
            # 背景が濃いセルは白字、薄いセルは黒字にして読みやすくする。
            tone = abs(v - (vmin + vmax) / 2) / (span / 2)
            ax.text(
                c,
                r,
                format(v, fmt),
                ha="center",
                va="center",
                fontsize=7,
                color="white" if tone > 0.6 else "black",
            )
    return im


def plot_archetype_heatmap(archetypes: dict[str, list[float]], axes_labels: list[str]):
    """アーキタイプ（型）× 7 軸の定義（+1 好む / −1 嫌う / 0 中立）をヒートマップで描く。"""
    plt = _setup_plt()
    names = list(archetypes.keys())
    matrix = [archetypes[n] for n in names]
    fig, ax = plt.subplots(figsize=(8, 0.7 * len(names) + 1.5))
    im = _heatmap(ax, matrix, names, axes_labels, vmin=-1.0, vmax=1.0, cmap="RdBu_r", fmt="+.0f")
    ax.set_title("アーキタイプ × 嗜好軸（嗜好空間の基底ベクトル）", fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="嗜好（+好む / −嫌う）")
    fig.tight_layout()
    return fig


def plot_persona_axes_heatmap(persona_pref: dict[str, list[float]], axes_labels: list[str]):
    """ペルソナ × 7 軸の嗜好埋め込み θ（連続値）をヒートマップで描く（全ペルソナ俯瞰）。"""
    plt = _setup_plt()
    names = list(persona_pref.keys())
    matrix = [persona_pref[n] for n in names]
    fig, ax = plt.subplots(figsize=(8, 0.7 * len(names) + 1.5))
    im = _heatmap(ax, matrix, names, axes_labels, vmin=-1.0, vmax=1.0, cmap="RdBu_r", fmt="+.2f")
    ax.set_title("ペルソナ × 嗜好軸（θ = アーキタイプの凸結合）", fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="嗜好強度")
    fig.tight_layout()
    return fig


def plot_dataset_stats(persona_counts: dict[str, int], split: str):
    """ペルソナ別データ件数の棒グラフ（argmax ラベルによる不均衡を可視化）。"""
    plt = _setup_plt()
    names = list(persona_counts.keys())
    counts = [persona_counts[n] for n in names]
    total = sum(counts)
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(names))
    bars = ax.bar(x, counts, color=COLOR_FT, alpha=0.85)
    if total > 0:
        mean = total / len(counts)
        ax.axhline(mean, color="#555555", linestyle="--", linewidth=1.0, label=f"平均 {mean:.1f}")
        ax.legend(fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("件数")
    ax.set_title(
        f"ペルソナ別データ件数（{split} / 合計 {total}）— argmax ラベルの不均衡",
        fontsize=11,
        fontweight="bold",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, c in zip(bars, counts, strict=True):
        ax.annotate(
            str(c),
            xy=(bar.get_x() + bar.get_width() / 2, c),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    return fig


def plot_pipeline(
    persona: str,
    axes_labels: list[str],
    theta: list[float],
    attrs: list[int],
    fragments: list[str],
    prompt: str,
    personas: list[str],
    appeals: list[float],
    winner: str,
):
    """嗜好空間 → 属性サンプル → 語片 → プロンプト → argmax ラベルの一連の流れを 1 枚で描く。

    交互作用により「生成元ペルソナ（``persona``）」と「最も好むペルソナ（``winner``）」が
    ズレうる様子（＝難候補・リランカーの伸びしろ）を可視化する。
    """
    plt = _setup_plt()
    fig = plt.figure(figsize=(11, 9))
    gs = fig.add_gridspec(5, 1, height_ratios=[1.3, 0.8, 0.9, 0.6, 1.6], hspace=0.7)

    # 段 1: 生成元ペルソナの嗜好埋め込み θ（横棒）
    ax1 = fig.add_subplot(gs[0])
    ax1.barh(axes_labels, theta, color=bar_colors(theta), alpha=0.85)
    ax1.axvline(0, color="black", linewidth=0.8)
    ax1.set_xlim(-1.3, 1.3)
    ax1.set_title(f"① 嗜好空間: 生成元ペルソナ {persona} の θ", fontsize=10, fontweight="bold")
    ax1.tick_params(labelsize=8)

    # 段 2: 嗜好分布からサンプルした二値属性（0/1 のチップ）
    ax2 = fig.add_subplot(gs[1])
    ax2.set_xlim(0, len(axes_labels))
    ax2.set_ylim(0, 1)
    ax2.axis("off")
    ax2.set_title("② 属性サンプル: P(a=1)=σ(sharpness·θ)", fontsize=10, fontweight="bold")
    for i, (ax_name, a) in enumerate(zip(axes_labels, attrs, strict=True)):
        ax2.add_patch(
            plt.Rectangle(
                (i + 0.05, 0.1),
                0.9,
                0.8,
                facecolor=COLOR_FT if a == 1 else "#e8e8e8",
                edgecolor="#888888",
            )
        )
        ax2.text(i + 0.5, 0.5, str(a), ha="center", va="center", fontsize=11, fontweight="bold")
        ax2.text(i + 0.5, -0.15, ax_name, ha="center", va="top", fontsize=7, rotation=20)

    # 段 3: 属性 → 語片（FRAGMENTS）
    ax3 = fig.add_subplot(gs[2])
    ax3.axis("off")
    ax3.set_title("③ 語片化: attributes_to_fragments", fontsize=10, fontweight="bold")
    frag_text = "\n".join(
        f"  {ax_name:11s} = {frag}" for ax_name, frag in zip(axes_labels, fragments, strict=True)
    )
    ax3.text(0.0, 0.95, frag_text, ha="left", va="top", fontsize=8, family="monospace")

    # 段 4: 組み上がったプロンプト文字列
    ax4 = fig.add_subplot(gs[3])
    ax4.axis("off")
    ax4.set_title("④ プロンプト合成", fontsize=10, fontweight="bold")
    wrapped = "\n".join(textwrap.wrap(prompt, width=90))
    ax4.text(
        0.0,
        0.8,
        wrapped,
        ha="left",
        va="top",
        fontsize=9,
        family="monospace",
        bbox={"boxstyle": "round", "facecolor": "#f5f5f5", "edgecolor": "#bbbbbb"},
    )

    # 段 5: 全ペルソナの appeal（argmax で勝者＝ラベル）
    ax5 = fig.add_subplot(gs[4])
    colors = [COLOR_HIT if p == winner else COLOR_BASE for p in personas]
    ax5.barh(personas, appeals, color=colors, alpha=0.85)
    ax5.axvline(0, color="black", linewidth=0.6)
    note = "（生成元と一致）" if winner == persona else f"（生成元 {persona} と相違 → 難候補）"
    ax5.set_title(f"⑤ argmax appeal → ラベル = {winner} {note}", fontsize=10, fontweight="bold")
    ax5.set_xlabel("appeal（魅力度スコア）", fontsize=9)
    ax5.tick_params(labelsize=8)
    ax5.invert_yaxis()

    fig.suptitle("嗜好空間 → 生成プロンプト → argmax ラベル付け", fontsize=13, fontweight="bold")
    return fig


def plot_interaction_graph(
    persona: str, axes_labels: list[str], edges: list[tuple[int, int, float]], gamma: float
):
    """1 ペルソナの非加法的交互作用を、軸ノードを円環に置いたグラフで描く。

    緑エッジ＝両方 1 で「好む」加点、赤エッジ＝両方 1 で「嫌う」減点。線幅は |coef| に比例。
    """
    plt = _setup_plt()
    n = len(axes_labels)
    angles = [2 * np.pi * k / n + np.pi / 2 for k in range(n)]
    xs = [np.cos(a) for a in angles]
    ys = [np.sin(a) for a in angles]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.45, 1.45)
    ax.set_aspect("equal")
    ax.axis("off")

    max_coef = max((abs(c) for *_, c in edges), default=1.0) or 1.0
    for i, j, coef in edges:
        color = COLOR_HIT if coef > 0 else "#d62728"
        ax.plot(
            [xs[i], xs[j]],
            [ys[i], ys[j]],
            color=color,
            linewidth=1.5 + 3.5 * abs(coef) / max_coef,
            alpha=0.8,
            zorder=1,
        )
        mx, my = (xs[i] + xs[j]) / 2, (ys[i] + ys[j]) / 2
        ax.text(
            mx,
            my,
            f"{coef:+.1f}",
            ha="center",
            va="center",
            fontsize=9,
            color=color,
            fontweight="bold",
            bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "none", "alpha": 0.8},
            zorder=3,
        )

    for x, y, name in zip(xs, ys, axes_labels, strict=True):
        ax.scatter([x], [y], s=420, color="#dddddd", edgecolors="#666666", zorder=2)
        ax.text(x, y, name, ha="center", va="center", fontsize=8, zorder=4)

    ax.set_title(
        f"{persona} の非加法的交互作用\nappeal += γ({gamma:.1f}) × coef × (a_i AND a_j)"
        "   緑=好む / 赤=嫌う",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def plot_appeal_decomposition(
    personas: list[str],
    linear: list[float],
    interaction: list[float],
    popularity: list[float],
    winner: str,
    subtitle: str = "",
):
    """ある候補に対する各ペルソナの appeal を、寄与（線形/交互作用/人気）の積み上げ棒で示す。

    argmax（合計が最大）で勝者ペルソナ＝ラベルが決まる仕組みを定量的に見せる。
    """
    plt = _setup_plt()
    x = np.arange(len(personas))
    lin = np.array(linear)
    inter = np.array(interaction)
    pop = np.array(popularity)

    fig, ax = plt.subplots(figsize=(10, 5))
    # 正負が混ざるので、正の積み上げ・負の積み上げを分けて base を計算する。
    parts = [
        ("線形 θ·(2a−1)", lin, COLOR_BASE),
        ("交互作用 γ·Σcoef·AND", inter, COLOR_FT),
        ("人気バイアス λ·θ_g·(2a−1)", pop, "#999999"),
    ]
    pos_base = np.zeros(len(personas))
    neg_base = np.zeros(len(personas))
    for label, vals, color in parts:
        base = np.where(vals >= 0, pos_base, neg_base)
        ax.bar(x, vals, bottom=base, color=color, alpha=0.85, label=label)
        pos_base = pos_base + np.where(vals >= 0, vals, 0.0)
        neg_base = neg_base + np.where(vals < 0, vals, 0.0)

    totals = lin + inter + pop
    ax.scatter(x, totals, color="black", zorder=5, marker="D", s=30, label="合計 appeal")

    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    labels = [f"★{p}" if p == winner else p for p in personas]
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("appeal 寄与")
    title = "appeal の寄与分解（★ = argmax 勝者＝ラベル）"
    if subtitle:
        title += f"\n{subtitle}"
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")  # 中央の勝者バーと重ならない位置へ
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2 段階検索の効果（rerank_examples.json）
# ---------------------------------------------------------------------------


def plot_rank_changes(items: list[dict], top_k: int):
    """各クエリの正解画像の順位が rerank 前→後でどう動くかをスロープグラフで描く。

    ``items`` の各要素は ``{"query", "before", "after"}``（順位は 1 始まり、圏外は ``None``）。
    順位は小さいほど上位なので y 軸を反転し、圏外は最下段（``top_k + 1``）に置く。
    """
    if not items:
        return placeholder_figure("rerank_examples.json が見つかりません")

    plt = _setup_plt()
    fig, ax = plt.subplots(figsize=(7, 5))
    out_rank = top_k + 1  # 圏外（None）を置く擬似順位

    def _y(rank):
        return out_rank if rank is None else rank

    for it in items:
        b, a = _y(it.get("before")), _y(it.get("after"))
        if a < b:
            color = COLOR_HIT  # 順位が上がった＝改善
        elif a > b:
            color = "#d62728"  # 悪化
        else:
            color = "#888888"  # 変化なし
        ax.plot([0, 1], [b, a], color=color, marker="o", linewidth=2, alpha=0.8)
        ax.annotate(
            it.get("query", ""),
            xy=(0, b),
            xytext=(-6, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=8,
        )
        ax.annotate(
            "圏外" if it.get("after") is None else str(it.get("after")),
            xy=(1, a),
            xytext=(6, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8,
            color=color,
        )

    ax.set_xlim(-0.4, 1.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Rerank 前\n(埋め込み検索)", "Rerank 後\n(2 段階検索)"], fontsize=9)
    ax.set_ylim(out_rank + 0.5, 0.5)  # 反転（上位＝上）
    ax.set_ylabel("正解の順位（小さいほど上位）")
    ax.set_yticks(list(range(1, out_rank + 1)))
    ax.set_yticklabels([str(r) for r in range(1, out_rank)] + ["圏外"], fontsize=8)
    ax.set_title("リランクによる正解順位の変化（緑=改善）", fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_confusion_matrices(matrices: list, labels: list[str], titles: list[str]):
    """1 つ以上の混同行列（クエリ persona × 取得 persona）を横に並べて描く。

    対角が濃いほど「クエリと同じペルソナの画像を取れている」＝精度が高い。base と FT を
    並べると、FT で対角が濃くなる（混同が減る）様子が見える。
    """
    plt = _setup_plt()
    n = len(matrices)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5.5), squeeze=False)
    for ax, matrix, title in zip(axes[0], matrices, titles, strict=True):
        mat = np.array(matrix, dtype=float)
        vmax = float(mat.max()) if mat.size and mat.max() > 0 else 1.0
        im = _heatmap(
            ax, mat.tolist(), labels, labels, vmin=0.0, vmax=vmax, cmap="Blues", fmt=".0f"
        )
        ax.set_xlabel("取得された画像のペルソナ", fontsize=9)
        ax.set_ylabel("クエリのペルソナ", fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="件数")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 学習曲線（HF Trainer の trainer_state.json）
# ---------------------------------------------------------------------------


def parse_trainer_state(state: dict) -> dict:
    """``trainer_state.json`` の ``log_history`` を loss 系列と eval 系列に整形する（純関数）。

    Returns:
        ``{"loss": [(step, val), ...], "eval": {key: [(step, val), ...]}, "best_step": int|None}``
        loss 系列・eval 系列はいずれも step 昇順。
    """
    history = state.get("log_history", []) if state else []
    loss: list[tuple[int, float]] = []
    eval_series: dict[str, list[tuple[int, float]]] = {}
    for entry in history:
        step = entry.get("step")
        if step is None:
            continue
        if "loss" in entry:
            loss.append((step, float(entry["loss"])))
        for k, v in entry.items():
            if k.startswith("eval_") and isinstance(v, (int, float)):
                eval_series.setdefault(k, []).append((step, float(v)))
    loss.sort(key=lambda t: t[0])
    for series in eval_series.values():
        series.sort(key=lambda t: t[0])
    return {"loss": loss, "eval": eval_series, "best_step": state.get("best_global_step")}


def _select_eval_series(eval_series: dict, suffix: str) -> tuple[str, list] | tuple[None, None]:
    """eval 系列から、キーが ``suffix`` で終わるものを 1 つ選ぶ（無ければ (None, None)）。"""
    for key, series in eval_series.items():
        if key.endswith(suffix):
            return key, series
    return None, None


def plot_training_curve(curve: dict, title: str, eval_suffix: str = "ndcg@10"):
    """単一ステージの学習曲線（loss 左軸 ＋ eval 指標 右軸）を二軸で描く。"""
    loss = curve.get("loss", [])
    if not loss:
        return placeholder_figure("学習ログ（trainer_state.json）が見つかりません")

    plt = _setup_plt()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    steps = [s for s, _ in loss]
    vals = [v for _, v in loss]
    ax.plot(steps, vals, color=COLOR_BASE, marker=".", linewidth=1.5, label="train loss")
    ax.set_xlabel("step")
    ax.set_ylabel("train loss", color=COLOR_BASE)
    ax.tick_params(axis="y", labelcolor=COLOR_BASE)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    eval_key, eval_pts = _select_eval_series(curve.get("eval", {}), eval_suffix)
    if eval_pts:
        ax2 = ax.twinx()
        es = [s for s, _ in eval_pts]
        ev = [v for _, v in eval_pts]
        short = strip_prefix(eval_key[len("eval_") :]) if eval_key else eval_suffix
        ax2.plot(es, ev, color=COLOR_FT, marker="o", linewidth=1.8, label=f"eval {short}")
        ax2.set_ylabel(f"eval {short}", color=COLOR_FT)
        ax2.tick_params(axis="y", labelcolor=COLOR_FT)
        ax2.set_ylim(0, 1.02)

    best_step = curve.get("best_step")
    if best_step is not None:
        ax.axvline(best_step, color="#888888", linestyle=":", linewidth=1.2)
        ax.text(best_step, ax.get_ylim()[1], f"best={best_step}", fontsize=7, va="top", ha="left")

    ax.set_title(title, fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_loss_overview(curves: dict[str, list], title: str):
    """複数ステージの train loss を 1 枚に重ねた俯瞰図を描く（``{stage_label: [(step, loss)]}``）。"""
    series = {k: v for k, v in curves.items() if v}
    if not series:
        return placeholder_figure("学習ログ（trainer_state.json）が見つかりません")

    plt = _setup_plt()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for label, pts in series.items():
        steps = [s for s, _ in pts]
        vals = [v for _, v in pts]
        ax.plot(steps, vals, marker=".", linewidth=1.5, label=label)
    ax.set_xlabel("step")
    ax.set_ylabel("train loss")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig
