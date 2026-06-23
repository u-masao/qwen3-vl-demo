"""README 用のサンプル画像（図）を生成する。

OSS として公開するとき、テキストだけの README より「生成画像」や「検索結果が
どう変わるか」が一目で分かる図があるほうが伝わりやすい。このモジュールは、
``make all`` 等で作った成果物（``data/`` と ``outputs/model``）から、次の 2 枚の
PNG を ``docs/images/`` に書き出す:

  * **生成画像グリッド** (``sample_grid.png``)
      生成済みデータセットの画像をカテゴリ横断でサンプリングし、キャプション付きの
      コンタクトシートにする。**モデル不要**（PIL / matplotlib のみ）なので CPU でも動く。

  * **検索 Before/After 図** (``retrieval_before_after.png``)
      同じクエリ（ペルソナ）に対して「ベース埋め込み」と「ファインチューニング済み
      埋め込み」がそれぞれ取ってくる上位画像を上下に並べ、正解（同一ペルソナ）の画像を
      緑枠でハイライトする。FT で正解が上位に上がる様子を可視化する。
      **埋め込みモデルのロードが必要**（GPU 推奨）。``cfg.model_path`` に FT 済みモデルが
      無い場合はスキップする。

検索ロジックは新規実装せず、``rerank.py`` の ``_retrieve_topk`` / ``_build_relevant`` を
そのまま再利用する（評価本体と完全に同じ取得規則で図を作るため）。

Gradio 画面のスクリーンショットはこのスクリプトでは作れない（ブラウザ撮影が必要）。
撮り方は ``docs/images/README.md`` を参照。
"""

from __future__ import annotations

import argparse
import json
import logging
import textwrap
from pathlib import Path

from datasets import load_from_disk

from . import plots
from .config import REPO_ROOT, Config, add_config_args, config_from_args

logger = logging.getLogger(__name__)

# 既定の出力先（リポジトリルート基準）。README がここを参照する。
DEFAULT_OUT_DIR = "docs/images"


def _resolve_out_dir(out_dir: str) -> Path:
    """出力ディレクトリを絶対パスに解決する（相対ならリポジトリルート基準）。"""
    path = Path(out_dir)
    path = path if path.is_absolute() else REPO_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_fig(fig, out_path: Path) -> Path:
    """Figure を PNG に保存して閉じ、パスを返す（新規 build_* の共通処理）。

    描画自体は plots.py に集約してあるので、figures 側はデータを集めて plots を呼び、
    結果の Figure をここで保存するだけにする。dpi / bbox は既存 2 図に合わせる。
    """
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plots._setup_plt().close(fig)  # pyplot はシングルトンなので同じインスタンスで閉じられる
    return out_path


def _select_grid_indices(ds, n: int) -> list[int]:
    """カテゴリを跨いでなるべく均等に ``n`` 件のインデックスを選ぶ。

    カテゴリごとに先頭から拾い、ラウンドロビンで集めることで、1 枚の図に多様な
    被写体（animal / vehicle / food / scene / object）が並ぶようにする。
    """
    by_cat: dict[str, list[int]] = {}
    for i, row in enumerate(ds):
        by_cat.setdefault(row["category"], []).append(i)

    picked: list[int] = []
    # 各カテゴリの「k 番目」を順番に拾うラウンドロビン。
    for rank in range(max((len(v) for v in by_cat.values()), default=0)):
        for cat in sorted(by_cat):
            if rank < len(by_cat[cat]):
                picked.append(by_cat[cat][rank])
                if len(picked) >= n:
                    return picked
    return picked


def build_sample_grid(cfg: Config, split: str, n: int, out_dir: Path) -> Path | None:
    """生成画像グリッド（コンタクトシート）を作って PNG に保存する。

    Args:
        cfg: 全体設定（``data_path`` を参照）。
        split: 読み込むスプリット（"train" / "eval"）。
        n: 並べる画像枚数（4 列グリッド）。
        out_dir: 出力ディレクトリ（絶対パス）。

    Returns:
        書き出した PNG のパス。データが無い場合は ``None``。
    """
    ds_path = cfg.data_path / split
    if not ds_path.exists():
        logger.warning(
            "データセットが見つかりません: %s（先に `make data` 等を実行してください）", ds_path
        )
        return None

    ds = load_from_disk(str(ds_path))
    indices = _select_grid_indices(ds, n)
    if not indices:
        logger.warning("グリッドに並べる画像がありません: %s", ds_path)
        return None

    plt = plots._setup_plt()
    cols = 4
    rows = (len(indices) + cols - 1) // cols
    # 行の高さは、画像の下に添えるキャプションが折り返しても次の行の画像に重ならないよう広めに取る。
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 3.3))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, idx in zip(axes, indices, strict=False):
        row = ds[idx]
        ax.imshow(row["positive"])
        # キャプション（anchor）は長いので折り返して画像の「下」に小さく添える。
        # 折り返し幅は画像セル幅に近づけ、1 行をできるだけ使う（文字サイズは据え置き）。
        caption = "\n".join(textwrap.wrap(row["anchor"], width=42))
        ax.set_xlabel(caption, fontsize=7)
        # 画像なので目盛り・枠線は消すが、xlabel（キャプション）は下に残す。
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    # 余った枠は消す。
    for ax in axes[len(indices) :]:
        ax.axis("off")

    fig.suptitle(f"Synthetic dataset samples ({split})", fontsize=12)
    # h_pad で行同士の縦間隔を確保し、下に添えたキャプションが次の行の画像に重ならないようにする。
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=3.0)
    out_path = out_dir / "sample_grid.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("生成画像グリッドを書き出しました -> %s", out_path)
    return out_path


def _pick_query_indices(ds, num_queries: int) -> list[int]:
    """表示用に、ペルソナが重複しないクエリのインデックスを ``num_queries`` 件選ぶ。"""
    seen: set[str] = set()
    picked: list[int] = []
    for i, row in enumerate(ds):
        if row["persona"] in seen:
            continue
        seen.add(row["persona"])
        picked.append(i)
        if len(picked) >= num_queries:
            break
    return picked


def build_retrieval_before_after(
    cfg: Config, num_queries: int, top_k: int, out_dir: Path
) -> Path | None:
    """検索 Before/After 図（ベース vs FT 埋め込み）を作って PNG に保存する。

    各クエリ（ペルソナ）について、ベース埋め込みと FT 埋め込みが取得する上位
    ``top_k`` 画像を 2 段に並べ、正解（同一ペルソナ）の画像を緑枠で示す。

    Returns:
        書き出した PNG のパス。前提が揃わない（eval が無い / FT モデルが無い）場合は ``None``。
    """
    # 検索の取得規則・正解定義は評価本体と完全に同じものを使う。
    from .prompts import PERSONA_MAP
    from .rerank import _build_relevant, _retrieve_topk

    # preference タスクでは「好み」は被写体ではなく見た目の属性なので、図のラベルは
    # 嗜好モデル（persona ごとの選好ベクトル）から復元する。subject タスクでは従来どおり
    # PERSONA_MAP の被写体名を使う。
    pref_model = None
    if cfg.data.task == "preference":
        from .preference import load_model

        pref_path = cfg.data_path / "preference_model.json"
        if pref_path.exists():
            pref_model = load_model(pref_path)
        else:
            logger.warning(
                "preference タスクですが嗜好モデルが見つかりません: %s。"
                "ラベルは被写体名にフォールバックします。",
                pref_path,
            )

    def _persona_caption(persona: str, n: int) -> str:
        """図の 1 段目に出す「このペルソナの好み」の 1 行ラベルを作る。"""
        if pref_model is not None:
            from .preference import persona_preferred_fragments

            # 属性語片はカンマを含む（例: "ornate, intricately detailed"）ため、
            # 区切りは中黒にして語片内のカンマと混同しないようにする。
            frags = persona_preferred_fragments(pref_model, persona, top_k=n)
            return f"{persona} : prefers → " + " ・ ".join(frags)
        items = PERSONA_MAP.get(persona, [])[:n]
        return f"{persona} : favorite → {', '.join(items)}"

    eval_path = cfg.data_path / "eval"
    if not eval_path.exists():
        logger.warning("eval データが見つかりません: %s（Before/After 図はスキップ）", eval_path)
        return None
    if not cfg.model_path.exists():
        logger.warning(
            "FT 済み埋め込みモデルが見つかりません: %s。"
            "Before/After 図には学習済みモデルが必要です（`make train` 後に再実行してください）。スキップします。",
            cfg.model_path,
        )
        return None

    eval_ds = load_from_disk(str(eval_path))
    corpus_images = [row["positive"] for row in eval_ds]
    queries = [row["persona"] for row in eval_ds]
    relevant = _build_relevant(eval_ds, cfg.data.relevant_same_category)

    k = min(top_k, len(corpus_images))
    # ラベルに出す「好み」の個数。preference は語片が長い（"ornate, intricately
    # detailed" など）ので、こだわりの強い上位の軸だけに絞る。
    caption_n = 3 if pref_model is not None else k
    show_idx = _pick_query_indices(eval_ds, num_queries)
    if not show_idx:
        logger.warning("表示できるクエリがありません。Before/After 図はスキップします。")
        return None

    logger.info("Before/After 図: base 埋め込みで取得")
    base_ranked = _retrieve_topk(cfg, cfg.embedding.model_id, queries, corpus_images, k)
    logger.info("Before/After 図: FT 埋め込みで取得")
    ft_ranked = _retrieve_topk(cfg, str(cfg.model_path), queries, corpus_images, k)

    from matplotlib.patches import Rectangle

    plt = plots._setup_plt()
    n_blocks = len(show_idx)
    # 1 ペルソナ = 3 段（好み一覧テキスト / Base 画像 / Fine-Tuned 画像）のブロック。
    # ブロック間にはスペーサー行を挟み、ブロック内より広い間隔を空ける
    # （枠線どうしが重ならないように）。
    text_h = 0.3  # テキスト段は画像段より低くする。
    spacer_h = 0.45  # ブロック間スペーサーの高さ。
    height_ratios: list[float] = []
    block_row0: list[int] = []  # 各ブロックの先頭（テキスト段）の行インデックス。
    for b in range(n_blocks):
        if b > 0:
            height_ratios.append(spacer_h)
        block_row0.append(len(height_ratios))
        height_ratios += [text_h, 1.0, 1.0]
    fig_h = sum(height_ratios) * 1.9
    fig = plt.figure(figsize=(k * 1.8, fig_h))
    gs = fig.add_gridspec(
        len(height_ratios), k, height_ratios=height_ratios, hspace=0.1, wspace=0.06
    )

    # 後でブロックを囲む枠線・段ラベルを描くために、各段の axes を覚えておく。
    block_geom: list[dict] = []
    for b, qi in enumerate(show_idx):
        rel = relevant[qi]
        r0 = block_row0[b]

        # 1 段目: 「user_xxx : prefers → …（好む属性）」を 1 行のテキストで。
        # preference タスクは見た目の属性、subject タスクは好む被写体名を出す。
        ax_text = fig.add_subplot(gs[r0, :])
        ax_text.axis("off")
        # 画像の左端ではなく、枠内に確保したラベル帯のぶんだけ右に寄せて書き出す。
        ax_text.text(
            0.0,
            0.5,
            _persona_caption(queries[qi], caption_n),
            ha="left",
            va="center",
            fontsize=12,
            color="#2ca02c",
            transform=ax_text.transAxes,
        )

        # 2・3 段目: Base / Fine-Tuned の取得画像を k 枚ずつ。
        row_axes: dict[str, list] = {}
        for which, (label, items) in enumerate(
            (("Base model", base_ranked[qi]), ("Fine-Tuned model", ft_ranked[qi]))
        ):
            r = r0 + 1 + which
            row_axes[label] = []
            for c in range(k):
                ax = fig.add_subplot(gs[r, c])
                row_axes[label].append(ax)
                ax.set_xticks([])
                ax.set_yticks([])
                if c >= len(items):
                    ax.axis("off")
                    continue
                doc = items[c]
                ax.imshow(corpus_images[doc])
                hit = doc in rel
                # 正解は緑の太枠、それ以外は薄いグレー枠で「枠あり」に統一する。
                for spine in ax.spines.values():
                    spine.set_edgecolor("#2ca02c" if hit else "#cccccc")
                    spine.set_linewidth(3.0 if hit else 0.8)
        block_geom.append({"text": ax_text, "rows": row_axes})

    fig.suptitle(
        "Personalized image retrieval — Base vs Fine-Tuned  (green = same persona as query)",
        fontsize=13,
    )
    # 枠線とラベルを描く前に余白を確定させる。左はラベル帯、右は枠線ぶんを空ける。
    fig.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.02)

    # 図全体を覆う透明オーバーレイに、ブロックの枠線と段ラベルを図座標で描く。
    overlay = fig.add_axes((0, 0, 1, 1))
    overlay.set_xlim(0, 1)
    overlay.set_ylim(0, 1)
    overlay.axis("off")
    overlay.set_navigate(False)
    # 枠の内側パディング。左は段ラベル帯ぶん広め、右は画像と枠線の間に隙間を作る。
    # 上下はブロック間スペーサーより小さくして、隣のブロックの枠と重ならないようにする。
    pad_l, pad_r, pad_t, pad_b = 0.075, 0.02, 0.008, 0.008
    for g in block_geom:
        boxes = [g["text"].get_position()]
        boxes += [ax.get_position() for axlist in g["rows"].values() for ax in axlist]
        x0 = min(p.x0 for p in boxes)
        x1 = max(p.x1 for p in boxes)
        y0 = min(p.y0 for p in boxes)
        y1 = max(p.y1 for p in boxes)
        overlay.add_patch(
            Rectangle(
                (x0 - pad_l, y0 - pad_b),
                (x1 - x0) + pad_l + pad_r,
                (y1 - y0) + pad_t + pad_b,
                fill=False,
                edgecolor="#888888",
                linewidth=1.2,
            )
        )
        # 段ラベル（Base model / Fine-Tuned model）を枠の内側・左の帯に縦書きで。
        for label, axlist in g["rows"].items():
            p = axlist[0].get_position()
            overlay.text(
                x0 - pad_l * 0.45,
                (p.y0 + p.y1) / 2,
                label,
                rotation=90,
                ha="center",
                va="center",
                fontsize=11,
            )
    out_path = out_dir / "retrieval_before_after.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("検索 Before/After 図を書き出しました -> %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# 嗜好モデル・メトリクスの読込ヘルパ（純ロジックは GPU 不要・テスト対象）
# ---------------------------------------------------------------------------


def _load_pref_dict(cfg: Config) -> dict | None:
    """preference_model.json を生 dict で読む（嗜好構造図の共通入力）。無ければ None。"""
    path = cfg.data_path / "preference_model.json"
    if not path.exists():
        logger.warning("嗜好モデルが見つかりません: %s（`make data` 後に再実行してください）", path)
        return None
    return json.loads(path.read_text())


def _load_pref_model_obj(cfg: Config):
    """preference_model.json を :class:`PreferenceModel` として読む（appeal 計算に使う）。"""
    path = cfg.data_path / "preference_model.json"
    if not path.exists():
        logger.warning("嗜好モデルが見つかりません: %s（`make data` 後に再実行してください）", path)
        return None
    from .preference import load_model

    return load_model(path)


def _count_personas(personas, order: list[str] | None = None) -> dict[str, int]:
    """ペルソナ列の出現数を数える（純関数）。

    ``order`` を与えるとその順序で並べ、欠席ペルソナは 0 件で埋める。``order`` が ``None`` の
    ときは件数の多い順。``order`` に無いペルソナが出てきた場合は末尾に足す（取りこぼし防止）。
    """
    counts: dict[str, int] = {}
    for p in personas:
        counts[p] = counts.get(p, 0) + 1
    if order is None:
        return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))
    result = {name: counts.get(name, 0) for name in order}
    for name, c in counts.items():
        if name not in result:
            result[name] = c
    return result


def _interaction_edges(model: dict, persona: str) -> list[tuple[int, int, float]]:
    """嗜好モデル dict から (axis_i, axis_j, coef) のリストを取り出す（純関数）。"""
    edges: list[tuple[int, int, float]] = []
    for tri in model.get("interactions", {}).get(persona, []):
        edges.append((int(tri[0]), int(tri[1]), float(tri[2])))
    return edges


def _appeal_components(model, persona: str, attrs: list[int]) -> tuple[float, float, float]:
    """appeal を線形項 / 交互作用項 / 人気バイアス項に分解する（ノイズ除く・純ロジック）。

    :func:`preference.appeal` の各項（line 235-239）と同じ式。``model`` は
    :class:`PreferenceModel`。ノイズは決定的だが説明上の意味が薄いので分解には含めない。
    """
    from .preference import _centered, _dot

    centered = _centered(attrs)
    linear = _dot(model.persona_pref[persona], centered)
    interaction = 0.0
    for tri in model.interactions.get(persona, []):
        i, j, c = int(tri[0]), int(tri[1]), tri[2]
        interaction += model.gamma * c * (attrs[i] * attrs[j])
    popularity = model.lam * _dot(model.global_pref, centered)
    return linear, interaction, popularity


def _select_latest_checkpoint(names: list[str]) -> str | None:
    """'checkpoint-<N>' 名のリストから最大 N のものを返す（純関数。無ければ None）。"""
    best_name, best_n = None, -1
    for name in names:
        suffix = name.rsplit("-", 1)[-1]
        if not suffix.isdigit():
            continue
        n = int(suffix)
        if n > best_n:
            best_name, best_n = name, n
    return best_name


def _latest_trainer_state(stage_dir: Path) -> dict | None:
    """``stage_dir`` 配下の最新 checkpoint から ``trainer_state.json`` を読む。無ければ None。"""
    if not stage_dir.exists():
        return None
    names = [p.name for p in stage_dir.glob("checkpoint-*") if p.is_dir()]
    latest = _select_latest_checkpoint(names)
    if latest is None:
        return None
    state_path = stage_dir / latest / "trainer_state.json"
    if not state_path.exists():
        return None
    return json.loads(state_path.read_text())


def _confusion_matrix(ranked, queries, doc_personas, order: list[str]) -> list[list[int]]:
    """検索結果から「クエリ persona × 取得 persona」の件数行列を組む（純ロジック）。"""
    index = {p: i for i, p in enumerate(order)}
    matrix = [[0] * len(order) for _ in order]
    for qi, q in enumerate(queries):
        for doc in ranked[qi]:
            matrix[index[q]][index[doc_personas[doc]]] += 1
    return matrix


# ---------------------------------------------------------------------------
# メトリクス比較図（JSON から。蒸留 variant は含めない）
# ---------------------------------------------------------------------------


def build_metrics_figure(cfg: Config, out_dir: Path) -> Path | None:
    """埋め込み Base vs FT のメトリクス比較棒グラフを書き出す（app タブ 1 と同一）。"""
    base_path = cfg.output_path / "metrics_base.json"
    ft_path = cfg.output_path / "metrics_finetuned.json"
    base = json.loads(base_path.read_text()) if base_path.exists() else {}
    ft = json.loads(ft_path.read_text()) if ft_path.exists() else {}
    if not base and not ft:
        logger.warning(
            "メトリクス JSON が見つかりません: %s / %s（先に評価を実行してください）",
            base_path,
            ft_path,
        )
        return None
    fig = plots.plot_metrics(base, ft, "Base vs Fine-tuned（埋め込み検索）")
    out_path = _save_fig(fig, out_dir / "metrics_base_vs_ft.png")
    logger.info("メトリクス比較図を書き出しました -> %s", out_path)
    return out_path


def build_rerank_metrics_figure(cfg: Config, out_dir: Path) -> Path | None:
    """2 段階検索の主要 4 パターン比較棒グラフを書き出す（app タブ 5 と同一・蒸留なし）。"""
    path = cfg.output_path / "rerank_metrics.json"
    if not path.exists():
        logger.warning("rerank_metrics.json が見つかりません: %s（rerank を先に実行）", path)
        return None
    metrics = json.loads(path.read_text())
    fig = plots.plot_rerank_metrics(
        metrics, "2 段階検索 主要 4 パターン比較（埋め込み × リランカー）"
    )
    out_path = _save_fig(fig, out_dir / "rerank_metrics.png")
    logger.info("2 段階検索比較図を書き出しました -> %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# 嗜好モデルの構造を説明する図（preference_model.json のみ・CPU 可）
# ---------------------------------------------------------------------------


def build_archetype_heatmap(cfg: Config, out_dir: Path) -> Path | None:
    """アーキタイプ × 軸の定義ヒートマップを書き出す。"""
    model = _load_pref_dict(cfg)
    if model is None:
        return None
    fig = plots.plot_archetype_heatmap(model["archetypes"], model["axes"])
    out_path = _save_fig(fig, out_dir / "preference_archetypes.png")
    logger.info("アーキタイプヒートマップを書き出しました -> %s", out_path)
    return out_path


def build_persona_embedding_grid(cfg: Config, out_dir: Path) -> Path | None:
    """全ペルソナ × 軸の嗜好 θ ヒートマップを書き出す（俯瞰版）。"""
    model = _load_pref_dict(cfg)
    if model is None:
        return None
    fig = plots.plot_persona_axes_heatmap(model["persona_pref"], model["axes"])
    out_path = _save_fig(fig, out_dir / "persona_embeddings.png")
    logger.info("ペルソナ嗜好ヒートマップを書き出しました -> %s", out_path)
    return out_path


def build_dataset_stats(cfg: Config, split: str, out_dir: Path) -> Path | None:
    """ペルソナ別データ件数（argmax ラベルの不均衡）の棒グラフを書き出す。"""
    ds_path = cfg.data_path / split
    if not ds_path.exists():
        logger.warning("データセットが見つかりません: %s（`make data` 後に再実行）", ds_path)
        return None
    ds = load_from_disk(str(ds_path))
    model = _load_pref_dict(cfg)
    order = list(model["persona_pref"].keys()) if model else None
    counts = _count_personas(ds["persona"], order)
    fig = plots.plot_dataset_stats(counts, split)
    out_path = _save_fig(fig, out_dir / "dataset_persona_counts.png")
    logger.info("データセット統計図を書き出しました -> %s", out_path)
    return out_path


def build_pipeline_figure(cfg: Config, out_dir: Path, persona: str | None = None) -> Path | None:
    """嗜好空間 → 属性 → 語片 → プロンプト → argmax ラベルのパイプライン図を書き出す。"""
    import random

    from .preference import (
        appeal,
        assign_persona,
        attributes_to_fragments,
        sample_item_attributes,
    )

    model = _load_pref_model_obj(cfg)
    if model is None:
        return None
    persona = persona or model.personas()[0]
    if persona not in model.persona_pref:
        logger.warning("ペルソナ %s が嗜好モデルにありません（パイプライン図はスキップ）", persona)
        return None
    rng = random.Random(cfg.seed)
    attrs = sample_item_attributes(model, persona, rng)
    fragments = attributes_to_fragments(model, attrs)
    subj = "cat"  # 被写体は属性と独立（プロンプト例示用に固定）
    prompt = f"a photo of a {subj}, " + ", ".join(fragments)
    personas = model.personas()
    appeals = [appeal(model, p, attrs) for p in personas]
    winner = assign_persona(model, attrs)
    fig = plots.plot_pipeline(
        persona,
        model.axes,
        model.persona_pref[persona],
        attrs,
        fragments,
        prompt,
        personas,
        appeals,
        winner,
    )
    out_path = _save_fig(fig, out_dir / "preference_pipeline.png")
    logger.info("嗜好パイプライン図を書き出しました -> %s", out_path)
    return out_path


def build_interaction_figure(cfg: Config, out_dir: Path, persona: str | None = None) -> Path | None:
    """非加法的交互作用を軸ノードの円環グラフで書き出す（既定は交互作用が最多のペルソナ）。"""
    model = _load_pref_dict(cfg)
    if model is None:
        return None
    interactions = model.get("interactions", {})
    if persona is None:
        persona = max(interactions, key=lambda p: len(interactions[p]), default=None)
    if not persona or not interactions.get(persona):
        logger.warning("交互作用を持つペルソナがありません（交互作用図はスキップ）")
        return None
    edges = _interaction_edges(model, persona)
    fig = plots.plot_interaction_graph(persona, model["axes"], edges, model.get("gamma", 2.0))
    out_path = _save_fig(fig, out_dir / "preference_interactions.png")
    logger.info("交互作用図を書き出しました -> %s", out_path)
    return out_path


def build_appeal_decomposition(
    cfg: Config, out_dir: Path, persona: str | None = None
) -> Path | None:
    """ある候補に対する各ペルソナの appeal を寄与分解した積み上げ棒を書き出す。"""
    import random

    from .preference import assign_persona, sample_item_attributes

    model = _load_pref_model_obj(cfg)
    if model is None:
        return None
    persona = persona or model.personas()[0]
    rng = random.Random(cfg.seed)
    attrs = sample_item_attributes(model, persona, rng)
    personas = model.personas()
    linear, interaction, popularity = [], [], []
    for p in personas:
        lin, inter, pop = _appeal_components(model, p, attrs)
        linear.append(lin)
        interaction.append(inter)
        popularity.append(pop)
    winner = assign_persona(model, attrs)
    subtitle = f"候補属性 = {persona} からのサンプル attrs={attrs}"
    fig = plots.plot_appeal_decomposition(
        personas, linear, interaction, popularity, winner, subtitle
    )
    out_path = _save_fig(fig, out_dir / "appeal_decomposition.png")
    logger.info("appeal 分解図を書き出しました -> %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# 2 段階検索の効果（rerank_examples.json / 検索の混同行列）
# ---------------------------------------------------------------------------


def build_rerank_rank_changes(cfg: Config, out_dir: Path) -> Path | None:
    """各クエリの正解順位が rerank 前→後でどう動くかのスロープグラフを書き出す。"""
    path = cfg.output_path / "rerank_examples.json"
    if not path.exists():
        logger.warning("rerank_examples.json が見つかりません: %s（rerank を先に実行）", path)
        return None
    examples = json.loads(path.read_text())
    items = [
        {
            "query": ex.get("query", ""),
            "before": ex.get("best_rank_before_rerank"),
            "after": ex.get("best_rank_after_rerank"),
        }
        for ex in examples
    ]
    top_k = max((ex.get("top_k", 10) for ex in examples), default=10)
    fig = plots.plot_rank_changes(items, top_k)
    out_path = _save_fig(fig, out_dir / "rerank_rank_changes.png")
    logger.info("リランク順位変化図を書き出しました -> %s", out_path)
    return out_path


def build_confusion_matrix(cfg: Config, out_dir: Path, top_k: int = 5) -> Path | None:
    """検索の混同行列（クエリ persona × 取得 persona）を base / FT 2 枚並べて書き出す。

    各ペルソナ名を 1 クエリとして上位 ``top_k`` を取り、取得画像の persona ラベルを集計する。
    埋め込みモデルのロードが必要（FT 済みモデルが無い場合はスキップ）。
    """
    from .rerank import _retrieve_topk

    eval_path = cfg.data_path / "eval"
    if not eval_path.exists():
        logger.warning("eval データが見つかりません: %s（混同行列はスキップ）", eval_path)
        return None
    if not cfg.model_path.exists():
        logger.warning(
            "FT 済み埋め込みモデルが見つかりません: %s（混同行列はスキップ）", cfg.model_path
        )
        return None

    eval_ds = load_from_disk(str(eval_path))
    corpus_images = [row["positive"] for row in eval_ds]
    doc_personas = [row["persona"] for row in eval_ds]
    order = sorted(set(doc_personas))
    k = min(top_k, len(corpus_images))

    logger.info("混同行列: base 埋め込みで取得")
    ranked_base = _retrieve_topk(cfg, cfg.embedding.model_id, order, corpus_images, k)
    logger.info("混同行列: FT 埋め込みで取得")
    ranked_ft = _retrieve_topk(cfg, str(cfg.model_path), order, corpus_images, k)

    mat_base = _confusion_matrix(ranked_base, order, doc_personas, order)
    mat_ft = _confusion_matrix(ranked_ft, order, doc_personas, order)
    fig = plots.plot_confusion_matrices(
        [mat_base, mat_ft],
        order,
        [f"Base 埋め込み（top-{k}）", f"Fine-tuned 埋め込み（top-{k}）"],
    )
    out_path = _save_fig(fig, out_dir / "retrieval_confusion.png")
    logger.info("検索の混同行列を書き出しました -> %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# 学習曲線（HF Trainer の trainer_state.json）
# ---------------------------------------------------------------------------


def build_training_curves(cfg: Config, out_dir: Path) -> Path | None:
    """埋め込み FT の学習曲線（loss ＋ eval NDCG@10）を書き出す。"""
    state = _latest_trainer_state(cfg.output_path / "checkpoints")
    if state is None:
        logger.warning(
            "学習ログ（checkpoints/trainer_state.json）が見つかりません（学習曲線はスキップ）"
        )
        return None
    curve = plots.parse_trainer_state(state)
    fig = plots.plot_training_curve(curve, "埋め込み FT の学習曲線（loss と eval NDCG@10）")
    out_path = _save_fig(fig, out_dir / "training_curve.png")
    logger.info("学習曲線を書き出しました -> %s", out_path)
    return out_path


def build_loss_overview(cfg: Config, out_dir: Path) -> Path | None:
    """3 ステージ（埋め込み / リランカー / 蒸留）の train loss を 1 枚に重ねた俯瞰図を書き出す。"""
    stages = {
        "embedding (train)": "checkpoints",
        "reranker": "reranker_checkpoints",
        "distill": "distill_checkpoints",
    }
    curves: dict[str, list] = {}
    for label, sub in stages.items():
        state = _latest_trainer_state(cfg.output_path / sub)
        if state is not None:
            curves[label] = plots.parse_trainer_state(state)["loss"]
    if not any(curves.values()):
        logger.warning("学習ログが見つかりません（loss 俯瞰図はスキップ）")
        return None
    fig = plots.plot_loss_overview(curves, "学習 loss の俯瞰（3 ステージ）")
    out_path = _save_fig(fig, out_dir / "training_loss_overview.png")
    logger.info("loss 俯瞰図を書き出しました -> %s", out_path)
    return out_path


def run_figures(
    cfg: Config,
    split: str = "eval",
    num_grid: int = 12,
    num_queries: int = 3,
    top_k: int = 5,
    out_dir: str = DEFAULT_OUT_DIR,
    pipeline_persona: str | None = None,
) -> None:
    """README 用の図をまとめて生成する（前提データが無い図は各々スキップ）。"""
    out = _resolve_out_dir(out_dir)
    # 生成画像グリッド・検索 Before/After（既存）
    build_sample_grid(cfg, split=split, n=num_grid, out_dir=out)
    build_retrieval_before_after(cfg, num_queries=num_queries, top_k=top_k, out_dir=out)
    # メトリクス比較（蒸留は含めない）
    build_metrics_figure(cfg, out)
    build_rerank_metrics_figure(cfg, out)
    # 嗜好モデルの構造図（preference_model.json のみ・CPU 可）
    build_archetype_heatmap(cfg, out)
    build_persona_embedding_grid(cfg, out)
    build_pipeline_figure(cfg, out, persona=pipeline_persona)
    build_interaction_figure(cfg, out)
    build_appeal_decomposition(cfg, out, persona=pipeline_persona)
    build_dataset_stats(cfg, split=split, out_dir=out)
    # 2 段階検索の効果
    build_rerank_rank_changes(cfg, out)
    build_confusion_matrix(cfg, out, top_k=top_k)
    # 学習曲線
    build_training_curves(cfg, out)
    build_loss_overview(cfg, out)


def main() -> None:
    """CLI エントリポイント: ``python -m qwen3vl_demo.figures`` / ``qwen3vl-figures``。"""
    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser(description="README 用のサンプル画像（図）を生成する。")
    add_config_args(parser)
    parser.add_argument(
        "--split", type=str, default="eval", help="グリッドに使うスプリット（既定: eval）。"
    )
    parser.add_argument(
        "--num-grid", type=int, default=12, help="グリッドに並べる画像枚数（既定: 12）。"
    )
    parser.add_argument(
        "--num-queries", type=int, default=3, help="Before/After 図に並べるクエリ数（既定: 3）。"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Before/After 図で 1 クエリあたり表示する上位件数（既定: 5）。",
    )
    parser.add_argument(
        "--out-dir", type=str, default=DEFAULT_OUT_DIR, help=f"出力先（既定: {DEFAULT_OUT_DIR}）。"
    )
    parser.add_argument(
        "--pipeline-persona",
        type=str,
        default=None,
        help="パイプライン図・appeal 分解図の代表ペルソナ（既定: 先頭ペルソナ）。",
    )
    args = parser.parse_args()
    cfg = config_from_args(args)
    run_figures(
        cfg,
        split=args.split,
        num_grid=args.num_grid,
        num_queries=args.num_queries,
        top_k=args.top_k,
        out_dir=args.out_dir,
        pipeline_persona=args.pipeline_persona,
    )


if __name__ == "__main__":
    main()
