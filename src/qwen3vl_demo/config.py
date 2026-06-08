"""設定の読み込み: YAML を入れ子の dataclass に変換し、共通の CLI 補助も提供する。

設計方針
--------
* 設定は **すべて YAML ファイル**（``configs/default.yaml`` / ``configs/smoke.yaml``）に集約し、
  コード中にマジックナンバーを散らさない。これにより DVC の ``params`` で設定差分を
  追跡でき、実験の再現性が保てる。
* YAML はセクションごとに dataclass へマッピングする。dataclass にすることで
  IDE 補完・型チェックが効き、``cfg.train.epochs`` のように安全にアクセスできる。
* すべてのエントリーポイント（generate_data / evaluate / train / rerank）は共通で
  ``--config PATH`` と ``--profile {default,smoke}`` を受け取る。``--profile`` は
  ``configs/<profile>.yaml`` を指すショートカット、``--config`` はその上書き。

パスの扱い
----------
YAML 内のパスはリポジトリルートからの相対パスとして書く。:class:`Config` の
``*_path`` プロパティが、リポジトリルート基準の **絶対パス** に解決して返すため、
どのカレントディレクトリから実行しても同じ場所を指す。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# このファイルは src/qwen3vl_demo/config.py にあるので、3 つ上がリポジトリルート。
# （parents[0]=qwen3vl_demo, parents[1]=src, parents[2]=リポジトリルート）
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


@dataclass
class Paths:
    """成果物の出力先ディレクトリ群（YAML の ``paths`` セクションに対応）。"""

    data_dir: str = "data"          # 生成したデータセット（train/eval）の保存先
    output_dir: str = "outputs"     # メトリクス JSON・リランク結果などの出力先
    model_dir: str = "outputs/model"  # ファインチューニング済みモデルの保存先


@dataclass
class DataCfg:
    """データ生成・評価まわりの設定（YAML の ``data`` セクション）。"""

    num_train: int = 200            # 学習用に生成する (caption, image) ペア数
    num_eval: int = 50              # 評価用に生成するペア数
    image_size: int = 512           # 生成画像の一辺ピクセル数（正方形）
    # True にすると、評価時に「同じカテゴリの画像」も正解とみなす（緩い評価）。
    # 既定の False はキャプションと画像の厳密な 1 対 1 対応のみを正解とする。
    relevant_same_category: bool = False


@dataclass
class ImageGenCfg:
    """画像生成モデル（SD-Turbo）の設定（YAML の ``image_gen`` セクション）。"""

    model_id: str = "stabilityai/sd-turbo"  # "stub" にすると合成スタブ画像に切替
    num_inference_steps: int = 1    # SD-Turbo は 1〜4 ステップ向けに蒸留されている
    guidance_scale: float = 0.0     # Turbo 系は classifier-free guidance を使わない
    batch_size: int = 8             # パイプライン 1 回あたりに生成する画像枚数


@dataclass
class EmbeddingCfg:
    """埋め込みモデル（Qwen3-VL-Embedding）の設定（YAML の ``embedding`` セクション）。"""

    model_id: str = "Qwen/Qwen3-VL-Embedding-2B"
    # Ada 世代では flash_attention_2 が使える。未導入なら models.py が自動で sdpa→既定へ
    # フォールバックするため、ここは「希望値」を書いておけばよい。
    attn_implementation: str = "flash_attention_2"
    # 画像 1 枚あたりのパッチ（トークン）数の上限。VRAM 節約のために大きい画像を抑制する。
    # None なら無制限（プロセッサのデフォルトに従う）。
    max_pixels: int | None = None
    # Qwen3-VL Embedding はクエリ側に instruction prompt を付ける運用がある。その名前。
    query_prompt_name: str | None = None


@dataclass
class RerankerCfg:
    """リランカー（Qwen3-VL-Reranker）の設定（YAML の ``reranker`` セクション）。"""

    model_id: str | None = "Qwen/Qwen3-VL-Reranker-2B"  # None ならリランク工程をスキップ
    top_k: int = 10                 # リランク対象とする検索上位件数
    # ファインチューニング済みリランカーの保存先。学習後ここに保存し、rerank 時に
    # 存在すれば優先して使う。
    model_dir: str = "outputs/reranker"
    # リランカー学習で 1 正例あたりに付与する負例（不一致ペア）の数。
    num_negatives: int = 3


@dataclass
class TrainCfg:
    """学習ハイパーパラメータ（YAML の ``train`` セクション）。"""

    epochs: int = 1
    per_device_batch_size: int = 4      # MNRL ではバッチが大きいほど負例が増えて有利
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2.0e-5
    warmup_ratio: float = 0.1
    gradient_checkpointing: bool = True  # VRAM 節約（速度と引き換え）
    eval_steps: int = 50                # 何ステップごとに評価器を回すか
    save_steps: int = 50
    logging_steps: int = 10


@dataclass
class Config:
    """全設定を束ねるトップレベルの設定オブジェクト。"""

    profile: str = "default"
    seed: int = 42
    device: str = "cuda"            # "cuda" or "cpu"
    dtype: str = "bfloat16"         # "float32" / "float16" / "bfloat16"
    paths: Paths = field(default_factory=Paths)
    data: DataCfg = field(default_factory=DataCfg)
    image_gen: ImageGenCfg = field(default_factory=ImageGenCfg)
    embedding: EmbeddingCfg = field(default_factory=EmbeddingCfg)
    reranker: RerankerCfg = field(default_factory=RerankerCfg)
    train: TrainCfg = field(default_factory=TrainCfg)

    # --- パスを絶対パスへ解決するアクセサ群 -----------------------------------
    def _abs(self, p: str) -> Path:
        """相対パスならリポジトリルート基準に、絶対パスならそのまま返す。"""
        path = Path(p)
        return path if path.is_absolute() else REPO_ROOT / path

    @property
    def data_path(self) -> Path:
        """データセット（train/eval）ディレクトリの絶対パス。"""
        return self._abs(self.paths.data_dir)

    @property
    def output_path(self) -> Path:
        """メトリクスなど出力ディレクトリの絶対パス。"""
        return self._abs(self.paths.output_dir)

    @property
    def model_path(self) -> Path:
        """ファインチューニング済み埋め込みモデル保存先の絶対パス。"""
        return self._abs(self.paths.model_dir)

    @property
    def reranker_model_path(self) -> Path:
        """ファインチューニング済みリランカー保存先の絶対パス。"""
        return self._abs(self.reranker.model_dir)

    @property
    def is_smoke(self) -> bool:
        """スモークプロファイル（CPU 配線確認）かどうか。"""
        return self.profile == "smoke"


def _build(section_cls, data: dict[str, Any] | None):
    """dict から dataclass を生成する。未知のキーは黙って無視する。

    YAML 側に dataclass が知らないキーがあってもエラーにせず、前方互換を保つための補助。
    """
    data = data or {}
    known = {f.name for f in section_cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    return section_cls(**{k: v for k, v in data.items() if k in known})


def load_config(path: str | Path) -> Config:
    """YAML 設定ファイルを読み込み :class:`Config` を返す。"""
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    # トップレベルのスカラ値はそのまま、各セクションは _build で dataclass 化する。
    return Config(
        profile=raw.get("profile", "default"),
        seed=raw.get("seed", 42),
        device=raw.get("device", "cuda"),
        dtype=raw.get("dtype", "bfloat16"),
        paths=_build(Paths, raw.get("paths")),
        data=_build(DataCfg, raw.get("data")),
        image_gen=_build(ImageGenCfg, raw.get("image_gen")),
        embedding=_build(EmbeddingCfg, raw.get("embedding")),
        reranker=_build(RerankerCfg, raw.get("reranker")),
        train=_build(TrainCfg, raw.get("train")),
    )


def add_config_args(parser: argparse.ArgumentParser) -> None:
    """共通の ``--config`` / ``--profile`` 引数をパーサに追加する。"""
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML 設定ファイルのパス。指定すると --profile より優先される。",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="default",
        choices=["default", "smoke"],
        help="configs/<profile>.yaml を選ぶショートカット（既定: default）。",
    )


def config_from_args(args: argparse.Namespace) -> Config:
    """パース済み引数から :class:`Config` を解決する（--config が --profile に優先）。"""
    if args.config:
        path = Path(args.config)
    else:
        path = CONFIG_DIR / f"{args.profile}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {path}")
    return load_config(path)


def resolve_dtype(name: str):
    """dtype 文字列を torch の dtype に変換する（torch は遅延 import）。

    torch をモジュールトップで import しないのは、CPU のみ・torch 未導入の環境でも
    config モジュール自体は import できるようにするため。
    """
    import torch

    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]
