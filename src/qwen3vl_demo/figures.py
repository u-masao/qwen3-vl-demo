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
import logging
import textwrap
from pathlib import Path

from datasets import load_from_disk

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


def _setup_matplotlib():
    """非表示（Agg）バックエンドで matplotlib を用意して plt を返す。

    matplotlib / japanize-matplotlib はここで遅延 import する。図を作らない経路
    （他モジュールからの import 等）で重い描画依存を読み込まないため。
    """
    import matplotlib

    matplotlib.use("Agg")  # ディスプレイの無い環境でも PNG を書き出せるように
    import matplotlib.pyplot as plt

    try:  # 日本語キャプションが将来入っても文字化けしないように（任意依存扱い）
        import japanize_matplotlib  # noqa: F401
    except Exception:  # noqa: BLE001 - 無くても英語キャプションなら問題ない
        pass
    return plt


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
        logger.warning("データセットが見つかりません: %s（先に `make data` 等を実行してください）", ds_path)
        return None

    ds = load_from_disk(str(ds_path))
    indices = _select_grid_indices(ds, n)
    if not indices:
        logger.warning("グリッドに並べる画像がありません: %s", ds_path)
        return None

    plt = _setup_matplotlib()
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

    plt = _setup_matplotlib()
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
    gs = fig.add_gridspec(len(height_ratios), k, height_ratios=height_ratios,
                          hspace=0.1, wspace=0.06)

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
            0.0, 0.5,
            _persona_caption(queries[qi], caption_n),
            ha="left", va="center", fontsize=12, color="#2ca02c",
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
        "Text→image retrieval: Base vs Fine-Tuned (green = correct persona)",
        fontsize=14,
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
        overlay.add_patch(Rectangle(
            (x0 - pad_l, y0 - pad_b),
            (x1 - x0) + pad_l + pad_r, (y1 - y0) + pad_t + pad_b,
            fill=False, edgecolor="#888888", linewidth=1.2,
        ))
        # 段ラベル（Base model / Fine-Tuned model）を枠の内側・左の帯に縦書きで。
        for label, axlist in g["rows"].items():
            p = axlist[0].get_position()
            overlay.text(
                x0 - pad_l * 0.45, (p.y0 + p.y1) / 2, label,
                rotation=90, ha="center", va="center", fontsize=11,
            )
    out_path = out_dir / "retrieval_before_after.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("検索 Before/After 図を書き出しました -> %s", out_path)
    return out_path


def run_figures(
    cfg: Config,
    split: str = "eval",
    num_grid: int = 12,
    num_queries: int = 3,
    top_k: int = 5,
    out_dir: str = DEFAULT_OUT_DIR,
) -> None:
    """README 用の図をまとめて生成する。"""
    out = _resolve_out_dir(out_dir)
    build_sample_grid(cfg, split=split, n=num_grid, out_dir=out)
    build_retrieval_before_after(cfg, num_queries=num_queries, top_k=top_k, out_dir=out)


def main() -> None:
    """CLI エントリポイント: ``python -m qwen3vl_demo.figures`` / ``qwen3vl-figures``。"""
    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser(description="README 用のサンプル画像（図）を生成する。")
    add_config_args(parser)
    parser.add_argument("--split", type=str, default="eval", help="グリッドに使うスプリット（既定: eval）。")
    parser.add_argument("--num-grid", type=int, default=12, help="グリッドに並べる画像枚数（既定: 12）。")
    parser.add_argument(
        "--num-queries", type=int, default=3, help="Before/After 図に並べるクエリ数（既定: 3）。"
    )
    parser.add_argument(
        "--top-k", type=int, default=5, help="Before/After 図で 1 クエリあたり表示する上位件数（既定: 5）。"
    )
    parser.add_argument(
        "--out-dir", type=str, default=DEFAULT_OUT_DIR, help=f"出力先（既定: {DEFAULT_OUT_DIR}）。"
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
    )


if __name__ == "__main__":
    main()
