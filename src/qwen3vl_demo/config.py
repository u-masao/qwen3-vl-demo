"""設定の読み込み: YAML を入れ子の dataclass に変換し、共通の CLI 補助も提供する。

設計方針
--------
* 設定は **すべて YAML（``params.yaml`` および ``params_<profile>.yaml``）** に集約し、
  コード中にマジックナンバーを散らさない。``params.yaml`` がパイプライン実行時の
  「有効プロファイル」で、``make use-default`` / ``make use-smoke`` /
  ``make use-flux`` が ``params_<profile>.yaml`` を ``params.yaml`` にコピーして切り替える。
* YAML は first-level キー（``common`` / ``data`` / ``image_gen`` / ``embedding`` /
  ``reranker`` / ``train``）でセクション分けし、それぞれ dataclass へマッピングする。
  dataclass にすることで IDE 補完・型チェックが効き、``cfg.train.epochs`` のように
  安全にアクセスできる。
* DVC パイプラインでは、各ステージの ``cmd`` が **自分が使う値だけ** を ``${...}``
  展開で CLI 引数として受け取る（:func:`add_*_args` のオーバーライド引数群）。
  値の変化が cmd 文字列に反映され、その値を使うステージだけが再実行される。
  そのため ``params.yaml`` / ``config.py`` を DVC の ``deps`` や ``params:`` に
  宣言する必要がない（Issue #8）。

設定の解決順序
--------------
:func:`config_from_args` は次の順で :class:`Config` を組み立てる。

1. ベース YAML を読む。``--config PATH`` 指定があればそれ、なければ ``--profile NAME``
   に対応する ``params_<NAME>.yaml``、それも無指定なら有効な ``params.yaml``。
2. CLI のオーバーライド引数（``--epochs`` 等、未指定なら無効）で個別の値を上書きする。

DVC のステージはすべての必要値を ``${...}`` で明示的に渡すため、ベース YAML の値は
完全に上書きされる。人手の実行（``make smoke`` 等）はオーバーライドを与えず、
ベース YAML の値をそのまま使う。

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

# 有効プロファイルの設定ファイル。``make use-<profile>`` がここへコピーする。
DEFAULT_PARAMS = REPO_ROOT / "params.yaml"

# 「オーバーライド未指定」を表すセンチネル。``None`` 自体が有効な上書き値（max_pixels=null
# などを明示的に None にしたいケース）なので、``default=None`` では「未指定」と区別できない。
_UNSET = object()


@dataclass
class Paths:
    """成果物の出力先ディレクトリ群（YAML の ``common.paths`` セクションに対応）。"""

    data_dir: str = "data"  # 生成したデータセット（train/eval）の保存先
    output_dir: str = "outputs"  # メトリクス JSON・リランク結果などの出力先
    model_dir: str = "outputs/model"  # ファインチューニング済みモデルの保存先


@dataclass
class DataCfg:
    """データ生成・評価まわりの設定（YAML の ``data`` セクション）。"""

    num_train: int = 200  # 学習用に生成する (caption, image) ペア数
    num_eval: int = 50  # 評価用に生成するペア数
    image_size: int = 512  # 生成画像の一辺ピクセル数（正方形）
    # True にすると、評価時に「同じカテゴリの画像」も正解とみなす（緩い評価）。
    # 既定の False はキャプションと画像の厳密な 1 対 1 対応のみを正解とする。
    relevant_same_category: bool = False
    # データ生成タスクの選択。既定は "preference"（人間の嗖好モデルで属性を生成し
    # argmax appeal をペルソナラベルにする＝リランカーの伸びしろがある本流）。
    # "subject" は legacy（subject の恣意的割当・二値 relevance）で opt-in 用。
    # どちらも同一スキーマを出力するため、評価・学習・リランクの下流は不変。
    task: str = "preference"


@dataclass
class PreferenceCfg:
    """嗖好モデル（``data.task = preference`` 用）の設定（YAML の ``preference`` セクション）。

    値は :func:`qwen3vl_demo.preference.build_model` の knob にそのまま渡る。``gamma`` が
    交互作用（＝リランカーの伸びしろ）の強度を決める中心的なノブ。
    """

    gamma: float = 2.0  # 交互作用強度（0=加法的＝リランカー伸びしろ≈0、大きいほど伸びしろ大）
    lam: float = 0.3  # 人気バイアス強度
    sigma: float = 0.1  # 個人ノイズ振幅（決定的）
    sharpness: float = 2.0  # 属性サンプリングの鋭さ（高いほど嗖好に忠実＝一貫性が強い）


@dataclass
class ImageGenCfg:
    """画像生成モデル（FLUX.2-klein-4B）の設定（YAML の ``image_gen`` セクション）。"""

    model_id: str = "black-forest-labs/FLUX.2-klein-4B"  # "stub" にすると合成スタブ画像に切替
    num_inference_steps: int = 4  # FLUX.2-klein は 4 ステップ向けに蒸留されている
    guidance_scale: float = 1.0  # FLUX.2-klein-4B の推奨値
    batch_size: int = 1  # パイプライン 1 回あたりに生成する画像枚数（VRAM 節約のため 1）
    # 生成画像のローカルキャッシュ。True なら同一入力の再生成をスキップする。
    cache_enabled: bool = True
    # キャッシュ保存先（環境ごと使い捨て）。.gitignore の .cache/ 配下なので VCS 追跡されない。
    cache_dir: str = ".cache/imggen"


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
    top_k: int = 10  # リランク対象とする検索上位件数
    # ファインチューニング済みリランカーの保存先。学習後ここに保存し、rerank 時に
    # 存在すれば優先して使う。
    model_dir: str = "outputs/reranker"
    # リランカー学習で 1 正例あたりに付与する負例（不一致ペア）の数。
    num_negatives: int = 3
    # 評価（リランク）時の画像 1 枚あたりパッチ（トークン）数の上限。VRAM 節約のために
    # 大きい画像を抑制する（埋め込みの embedding.max_pixels と同方針）。None なら無制限
    # （プロセッサのデフォルトに従う＝フル解像度）。Issue #11。
    max_pixels: int | None = None


@dataclass
class TrainCfg:
    """学習ハイパーパラメータ（YAML の ``train`` セクション）。"""

    epochs: int = 1
    per_device_batch_size: int = 4  # MNRL ではバッチが大きいほど負例が増えて有利
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2.0e-5
    warmup_ratio: float = 0.1
    gradient_checkpointing: bool = True  # VRAM 節約（速度と引き換え）
    eval_steps: int = 50  # 何ステップごとに評価器を回すか
    save_steps: int = 50
    logging_steps: int = 10


@dataclass
class DistillCfg:
    """知識蒸留（teacher → student 埋め込み）の設定（YAML の ``distill`` セクション）。

    student は常に埋め込み bi-encoder。teacher を 2 つから選べる:

    * ``reranker`` … FT 済みリランカー（cross-encoder）のスコアを teacher に、(query, pos,
      neg) のマージンを **MarginMSELoss** で蒸留する（パターン A: cross→bi）。
      ``reranker.model_id`` が null（smoke 等）のときはスキップする。
    * ``oracle`` … 嗖好モデル（``preference.py``＝正解の作り手）の連続 appeal を soft
      relevance に変換し、**CoSENTLoss** で蒸留する（パターン B）。teacher 推論コスト 0 で、
      埋め込みと preference_model.json だけで動くため CPU/smoke でも回せる。
    """

    teacher: str = "reranker"  # "reranker"（パターン A）/ "oracle"（パターン B）
    model_dir: str = "outputs/model_distilled"  # 蒸留済み student 埋め込みの保存先
    num_negatives: int = 3  # 1 クエリあたりの負例数（蒸留ペアの構築に使う）
    # oracle: appeal → soft relevance のシグモイド温度（大きいほどラベルが平坦＝緩い）。
    # reranker teacher では未使用。
    temperature: float = 1.0


@dataclass
class Config:
    """全設定を束ねるトップレベルの設定オブジェクト。"""

    profile: str = "default"
    seed: int = 42
    device: str = "cuda"  # "cuda" or "cpu"
    dtype: str = "bfloat16"  # "float32" / "float16" / "bfloat16"
    paths: Paths = field(default_factory=Paths)
    data: DataCfg = field(default_factory=DataCfg)
    preference: PreferenceCfg = field(default_factory=PreferenceCfg)
    image_gen: ImageGenCfg = field(default_factory=ImageGenCfg)
    embedding: EmbeddingCfg = field(default_factory=EmbeddingCfg)
    reranker: RerankerCfg = field(default_factory=RerankerCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    distill: DistillCfg = field(default_factory=DistillCfg)

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
    def image_cache_path(self) -> Path:
        """生成画像キャッシュディレクトリの絶対パス。"""
        return self._abs(self.image_gen.cache_dir)

    @property
    def reranker_model_path(self) -> Path:
        """ファインチューニング済みリランカー保存先の絶対パス。"""
        return self._abs(self.reranker.model_dir)

    @property
    def distill_model_path(self) -> Path:
        """蒸留済み student 埋め込み保存先の絶対パス。"""
        return self._abs(self.distill.model_dir)

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
    """YAML 設定ファイルを読み込み :class:`Config` を返す。

    YAML は first-level キー構成（``common`` にトップレベルのスカラと ``paths``、
    残りはセクションごと）を想定する。
    """
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    common = raw.get("common") or {}
    return Config(
        profile=common.get("profile", "default"),
        seed=common.get("seed", 42),
        device=common.get("device", "cuda"),
        dtype=common.get("dtype", "bfloat16"),
        paths=_build(Paths, common.get("paths")),
        data=_build(DataCfg, raw.get("data")),
        preference=_build(PreferenceCfg, raw.get("preference")),
        image_gen=_build(ImageGenCfg, raw.get("image_gen")),
        embedding=_build(EmbeddingCfg, raw.get("embedding")),
        reranker=_build(RerankerCfg, raw.get("reranker")),
        train=_build(TrainCfg, raw.get("train")),
        distill=_build(DistillCfg, raw.get("distill")),
    )


# --- CLI 引数 ---------------------------------------------------------------
#
# ベース選択（--config / --profile）と、セクション別のオーバーライド引数群に分ける。
# 各エントリポイントは自分が使うセクションの ``add_*_args`` だけを呼べばよい。DVC の
# ステージはそのセクションの値を ``${...}`` で渡す。:func:`config_from_args` がベース
# YAML を読み、与えられたオーバーライドだけを適用する。


def _nullable_int(s: str) -> int | None:
    """ "none" / "null" / "" を None に、その他を int にする（DVC の null 展開対策）。"""
    return None if s.strip().lower() in ("none", "null", "") else int(s)


def _nullable_str(s: str) -> str | None:
    """ "none" / "null" / "" を None に、その他をそのまま返す（DVC の null 展開対策）。"""
    return None if s.strip().lower() in ("none", "null", "") else s


def _parse_bool(s: str) -> bool:
    """真偽値を表す文字列を bool にする。

    boolean のオーバーライドは ``--flag true`` のように **値指定** で受け取る。こうすると
    DVC のステージ ``cmd`` で ``--gradient-checkpointing ${train.gradient_checkpointing}``
    のように ``${...}`` 展開した値（True / False）をそのまま渡せる。
    """
    v = s.strip().lower()
    if v in ("true", "1", "yes", "y", "on"):
        return True
    if v in ("false", "0", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"真偽値として解釈できません: {s!r}")


def add_config_args(parser: argparse.ArgumentParser) -> None:
    """ベース設定の選択引数（``--config`` / ``--profile``）を追加する。"""
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML 設定ファイルのパス。指定すると --profile より優先される。",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help=(
            "params_<profile>.yaml を選ぶショートカット（例: default / smoke / flux）。"
            "未指定なら有効な params.yaml を読む。"
        ),
    )


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """``common`` セクションのオーバーライド引数（seed / device / dtype / paths）。"""
    g = parser.add_argument_group("common overrides")
    g.add_argument("--seed", type=int, default=_UNSET, help="乱数シード（common.seed）。")
    g.add_argument("--device", type=str, default=_UNSET, help="cuda / cpu（common.device）。")
    g.add_argument(
        "--dtype", type=str, default=_UNSET, help="float32 / float16 / bfloat16（common.dtype）。"
    )
    g.add_argument(
        "--data-dir", type=str, default=_UNSET, help="データセット出力先（common.paths.data_dir）。"
    )
    g.add_argument(
        "--output-dir", type=str, default=_UNSET, help="出力先（common.paths.output_dir）。"
    )
    g.add_argument(
        "--model-dir",
        type=str,
        default=_UNSET,
        help="FT 済み埋め込みモデル保存先（common.paths.model_dir）。",
    )


def add_data_args(parser: argparse.ArgumentParser) -> None:
    """``data`` セクションのオーバーライド引数。"""
    g = parser.add_argument_group("data overrides")
    g.add_argument("--num-train", type=int, default=_UNSET, help="学習用ペア数（data.num_train）。")
    g.add_argument("--num-eval", type=int, default=_UNSET, help="評価用ペア数（data.num_eval）。")
    g.add_argument(
        "--image-size",
        type=int,
        default=_UNSET,
        help="生成画像の一辺ピクセル数（data.image_size）。",
    )
    g.add_argument(
        "--relevant-same-category",
        type=_parse_bool,
        default=_UNSET,
        metavar="BOOL",
        help="同カテゴリ画像も正解扱いにするか（data.relevant_same_category）。",
    )
    g.add_argument(
        "--task",
        type=str,
        default=_UNSET,
        help="データ生成タスク: subject（既存）/ preference（嗖好モデル）（data.task）。",
    )


def add_preference_args(parser: argparse.ArgumentParser) -> None:
    """``preference`` セクションのオーバーライド引数（``data.task = preference`` 用）。"""
    g = parser.add_argument_group("preference overrides")
    g.add_argument(
        "--pref-gamma", type=float, default=_UNSET, help="交互作用強度（preference.gamma）。"
    )
    g.add_argument(
        "--pref-lam", type=float, default=_UNSET, help="人気バイアス強度（preference.lam）。"
    )
    g.add_argument(
        "--pref-sigma", type=float, default=_UNSET, help="個人ノイズ振幅（preference.sigma）。"
    )
    g.add_argument(
        "--pref-sharpness",
        type=float,
        default=_UNSET,
        help="属性サンプリングの鋭さ（preference.sharpness）。",
    )


def add_image_gen_args(parser: argparse.ArgumentParser) -> None:
    """``image_gen`` セクションのオーバーライド引数。"""
    g = parser.add_argument_group("image_gen overrides")
    g.add_argument(
        "--image-model",
        type=str,
        default=_UNSET,
        help="画像生成モデル ID / 'stub'（image_gen.model_id）。",
    )
    g.add_argument(
        "--steps",
        type=int,
        default=_UNSET,
        help="推論ステップ数（image_gen.num_inference_steps）。",
    )
    g.add_argument(
        "--guidance-scale",
        type=float,
        default=_UNSET,
        help="ガイダンススケール（image_gen.guidance_scale）。",
    )
    g.add_argument(
        "--image-batch-size",
        type=int,
        default=_UNSET,
        help="生成バッチサイズ（image_gen.batch_size）。",
    )
    g.add_argument(
        "--image-cache",
        type=_parse_bool,
        default=_UNSET,
        metavar="BOOL",
        help="生成画像キャッシュの有効/無効（image_gen.cache_enabled）。",
    )
    g.add_argument(
        "--cache-dir",
        type=str,
        default=_UNSET,
        help="生成画像キャッシュ保存先（image_gen.cache_dir）。",
    )


def add_embedding_args(parser: argparse.ArgumentParser) -> None:
    """``embedding`` セクションのオーバーライド引数。"""
    g = parser.add_argument_group("embedding overrides")
    g.add_argument(
        "--embedding-model",
        type=str,
        default=_UNSET,
        help="埋め込みモデル ID（embedding.model_id）。",
    )
    g.add_argument(
        "--attn-impl",
        type=str,
        default=_UNSET,
        help="attention 実装（embedding.attn_implementation）。",
    )
    g.add_argument(
        "--max-pixels",
        type=_nullable_int,
        default=_UNSET,
        help="画像トークン上限 / none（embedding.max_pixels）。",
    )
    g.add_argument(
        "--query-prompt-name",
        type=_nullable_str,
        default=_UNSET,
        help="クエリ instruction 名 / none（embedding.query_prompt_name）。",
    )


def add_reranker_args(parser: argparse.ArgumentParser) -> None:
    """``reranker`` セクションのオーバーライド引数。"""
    g = parser.add_argument_group("reranker overrides")
    g.add_argument(
        "--reranker-model",
        type=_nullable_str,
        default=_UNSET,
        help="リランカーモデル ID / none（reranker.model_id）。",
    )
    g.add_argument(
        "--top-k", type=int, default=_UNSET, help="リランク対象の上位件数（reranker.top_k）。"
    )
    g.add_argument(
        "--reranker-dir",
        type=str,
        default=_UNSET,
        help="FT 済みリランカー保存先（reranker.model_dir）。",
    )
    g.add_argument(
        "--num-negatives",
        type=int,
        default=_UNSET,
        help="リランカー学習の負例数（reranker.num_negatives）。",
    )
    g.add_argument(
        "--reranker-max-pixels",
        type=_nullable_int,
        default=_UNSET,
        help="リランク時の画像トークン上限 / none（reranker.max_pixels）。",
    )


def add_distill_args(parser: argparse.ArgumentParser) -> None:
    """``distill`` セクションのオーバーライド引数（知識蒸留）。"""
    g = parser.add_argument_group("distill overrides")
    g.add_argument(
        "--distill-teacher",
        type=str,
        default=_UNSET,
        help="蒸留の teacher: reranker（パターン A）/ oracle（パターン B）（distill.teacher）。",
    )
    g.add_argument(
        "--distill-model-dir",
        type=str,
        default=_UNSET,
        help="蒸留済み student 埋め込みの保存先（distill.model_dir）。",
    )
    g.add_argument(
        "--distill-num-negatives",
        type=int,
        default=_UNSET,
        help="蒸留ペアの 1 クエリあたり負例数（distill.num_negatives）。",
    )
    g.add_argument(
        "--distill-temperature",
        type=float,
        default=_UNSET,
        help="oracle soft relevance の温度（distill.temperature）。",
    )


def add_train_args(parser: argparse.ArgumentParser) -> None:
    """``train`` セクションのオーバーライド引数。"""
    g = parser.add_argument_group("train overrides")
    g.add_argument("--epochs", type=int, default=_UNSET, help="エポック数（train.epochs）。")
    g.add_argument(
        "--batch-size",
        type=int,
        default=_UNSET,
        help="デバイスあたりバッチサイズ（train.per_device_batch_size）。",
    )
    g.add_argument(
        "--grad-accum",
        type=int,
        default=_UNSET,
        help="勾配累積ステップ（train.gradient_accumulation_steps）。",
    )
    g.add_argument("--lr", type=float, default=_UNSET, help="学習率（train.learning_rate）。")
    g.add_argument(
        "--warmup-ratio",
        type=float,
        default=_UNSET,
        help="ウォームアップ比率（train.warmup_ratio）。",
    )
    g.add_argument(
        "--gradient-checkpointing",
        type=_parse_bool,
        default=_UNSET,
        metavar="BOOL",
        help="勾配チェックポイントの有効/無効（train.gradient_checkpointing）。",
    )
    g.add_argument(
        "--eval-steps", type=int, default=_UNSET, help="評価間隔ステップ（train.eval_steps）。"
    )
    g.add_argument(
        "--save-steps", type=int, default=_UNSET, help="保存間隔ステップ（train.save_steps）。"
    )
    g.add_argument(
        "--logging-steps",
        type=int,
        default=_UNSET,
        help="ログ間隔ステップ（train.logging_steps）。",
    )


# オーバーライド引数の dest → (Config 上の親オブジェクトを返す関数, 属性名) の対応表。
def _override_targets(cfg: Config):
    return {
        "seed": (cfg, "seed"),
        "device": (cfg, "device"),
        "dtype": (cfg, "dtype"),
        "data_dir": (cfg.paths, "data_dir"),
        "output_dir": (cfg.paths, "output_dir"),
        "model_dir": (cfg.paths, "model_dir"),
        "num_train": (cfg.data, "num_train"),
        "num_eval": (cfg.data, "num_eval"),
        "image_size": (cfg.data, "image_size"),
        "relevant_same_category": (cfg.data, "relevant_same_category"),
        "task": (cfg.data, "task"),
        "pref_gamma": (cfg.preference, "gamma"),
        "pref_lam": (cfg.preference, "lam"),
        "pref_sigma": (cfg.preference, "sigma"),
        "pref_sharpness": (cfg.preference, "sharpness"),
        "image_model": (cfg.image_gen, "model_id"),
        "steps": (cfg.image_gen, "num_inference_steps"),
        "guidance_scale": (cfg.image_gen, "guidance_scale"),
        "image_batch_size": (cfg.image_gen, "batch_size"),
        "image_cache": (cfg.image_gen, "cache_enabled"),
        "cache_dir": (cfg.image_gen, "cache_dir"),
        "embedding_model": (cfg.embedding, "model_id"),
        "attn_impl": (cfg.embedding, "attn_implementation"),
        "max_pixels": (cfg.embedding, "max_pixels"),
        "query_prompt_name": (cfg.embedding, "query_prompt_name"),
        "reranker_model": (cfg.reranker, "model_id"),
        "top_k": (cfg.reranker, "top_k"),
        "reranker_dir": (cfg.reranker, "model_dir"),
        "num_negatives": (cfg.reranker, "num_negatives"),
        "reranker_max_pixels": (cfg.reranker, "max_pixels"),
        "distill_teacher": (cfg.distill, "teacher"),
        "distill_model_dir": (cfg.distill, "model_dir"),
        "distill_num_negatives": (cfg.distill, "num_negatives"),
        "distill_temperature": (cfg.distill, "temperature"),
        "epochs": (cfg.train, "epochs"),
        "batch_size": (cfg.train, "per_device_batch_size"),
        "grad_accum": (cfg.train, "gradient_accumulation_steps"),
        "lr": (cfg.train, "learning_rate"),
        "warmup_ratio": (cfg.train, "warmup_ratio"),
        "gradient_checkpointing": (cfg.train, "gradient_checkpointing"),
        "eval_steps": (cfg.train, "eval_steps"),
        "save_steps": (cfg.train, "save_steps"),
        "logging_steps": (cfg.train, "logging_steps"),
    }


def _apply_overrides(cfg: Config, args: argparse.Namespace) -> None:
    """args 上に存在し、かつ実際に指定されたオーバーライド引数を cfg に反映する。

    引数群（``add_*_args``）を呼んでいないエントリポイントでは該当 dest が存在しないため
    ``getattr(..., _UNSET)`` で安全に無視する。``--profile`` で指定したプロファイル名は
    cfg.profile を上書きする（is_smoke 判定やラベルに使う）。
    """
    profile = getattr(args, "profile", None)
    if profile:
        cfg.profile = profile

    for dest, (obj, attr) in _override_targets(cfg).items():
        val = getattr(args, dest, _UNSET)
        if val is not _UNSET:
            setattr(obj, attr, val)


def config_from_args(args: argparse.Namespace) -> Config:
    """パース済み引数から :class:`Config` を解決する。

    ベース YAML（``--config`` > ``--profile`` の ``params_<profile>.yaml`` > 有効な
    ``params.yaml``）を読み、CLI のオーバーライド引数を適用して返す。
    """
    if getattr(args, "config", None):
        path = Path(args.config)
    elif getattr(args, "profile", None):
        path = REPO_ROOT / f"params_{args.profile}.yaml"
    else:
        path = DEFAULT_PARAMS
    if not path.exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {path}")

    cfg = load_config(path)
    _apply_overrides(cfg, args)
    return cfg


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
