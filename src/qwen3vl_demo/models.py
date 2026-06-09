"""Sentence Transformers 埋め込みモデルのロード補助。

``train.py`` と ``evaluate.py`` がまったく同じ手順でモデルを構築できるよう、
ロード処理をここに 1 箇所に集約している。ここを通すことで、attention 実装の
フォールバックや dtype / プロセッサ設定の扱いが両者で食い違わないようにする。
"""

from __future__ import annotations

import logging

from .config import Config, resolve_dtype

logger = logging.getLogger(__name__)


def load_embedding_model(cfg: Config, model_id: str | None = None):
    """設定に従って :class:`SentenceTransformer` 埋め込みモデルをロードする。

    Args:
        cfg: 全体設定。device / dtype / attention 実装 / 画像トークン上限を参照する。
        model_id: ロードするモデル ID またはローカルパス。``None`` の場合は
            ``cfg.embedding.model_id`` を使う。評価時にディスク上のファインチューニング
            済みモデル（``cfg.model_path``）を読むために上書きする用途を想定。

    Returns:
        構築済みの ``SentenceTransformer``。

    Notes:
        flash-attn が未導入だと ``flash_attention_2`` 指定でロードが失敗するため、
        ``sdpa`` → さらにモデル既定、の順に段階的にフォールバックする。
    """
    from sentence_transformers import SentenceTransformer

    model_id = model_id or cfg.embedding.model_id

    model_kwargs: dict = {}  # transformers のモデル本体へ渡す引数
    processor_kwargs: dict = {}  # 画像プロセッサ（前処理）へ渡す引数

    # dtype は GPU でのみ意味を持つ。CPU / スモークでは float32 のままにして移植性を保つ。
    if cfg.device != "cpu":
        model_kwargs["torch_dtype"] = resolve_dtype(cfg.dtype)
    if cfg.embedding.attn_implementation:
        model_kwargs["attn_implementation"] = cfg.embedding.attn_implementation
    if cfg.embedding.max_pixels:
        # 画像 1 枚あたりのパッチ数の上限。大きい画像のトークンを抑えて VRAM を節約する。
        processor_kwargs["max_pixels"] = cfg.embedding.max_pixels

    def _build(mk: dict, pk: dict):
        """与えられた kwargs で SentenceTransformer を生成する内部ヘルパ。"""
        kwargs: dict = {"device": cfg.device}
        if mk:
            kwargs["model_kwargs"] = mk
        if pk:
            kwargs["processor_kwargs"] = pk
        return SentenceTransformer(model_id, **kwargs)

    try:
        # まずは希望どおりの設定（既定では flash_attention_2）でロードを試みる。
        model = _build(model_kwargs, processor_kwargs)
    except (ImportError, ValueError, RuntimeError) as exc:
        # flash-attn 未導入、またはプロセッサ引数が非対応 → sdpa で再試行する。
        if model_kwargs.get("attn_implementation") == "flash_attention_2":
            logger.warning("  flash_attention_2 が使えません（%s）。sdpa で再試行します", exc)
            model_kwargs["attn_implementation"] = "sdpa"
            try:
                model = _build(model_kwargs, processor_kwargs)
            except (ImportError, ValueError, RuntimeError) as exc2:
                # sdpa も失敗したら attention 指定を外し、モデル既定の実装に任せる。
                logger.warning("  sdpa も失敗しました（%s）。モデル既定の実装を使います", exc2)
                model_kwargs.pop("attn_implementation", None)
                model = _build(model_kwargs, processor_kwargs)
        else:
            # flash_attention_2 以外の理由での失敗はフォールバックせず、そのまま送出する。
            raise

    return model
