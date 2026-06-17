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

    # 希望の attn 実装（None も可）。フォールバックで局所的に差し替えるため控えておく。
    attn_impl = model_kwargs.get("attn_implementation")

    def _load(pk: dict):
        """flash → sdpa → モデル既定 と attn をフォールバックしつつロードする（pk は固定）。

        Returns:
            ``(model, used_attn_impl)``。``model_kwargs`` は変更せずローカルコピーで試す。
        """
        mk = dict(model_kwargs)
        try:
            return _build(mk, pk), mk.get("attn_implementation", "モデル既定")
        except (ImportError, ValueError, RuntimeError) as exc:
            if attn_impl != "flash_attention_2":
                # flash_attention_2 以外の理由での失敗は attn フォールバックせず送出する。
                raise
            logger.warning("  flash_attention_2 が使えません（%s）。sdpa で再試行します", exc)
            logger.warning("  flash-attn を有効にするには: uv sync --extra gpu")
            mk["attn_implementation"] = "sdpa"
            try:
                return _build(mk, pk), "sdpa"
            except (ImportError, ValueError, RuntimeError) as exc2:
                # sdpa も失敗したら attention 指定を外し、モデル既定の実装に任せる。
                logger.warning("  sdpa も失敗しました（%s）。モデル既定の実装を使います", exc2)
                mk.pop("attn_implementation", None)
                return _build(mk, pk), "モデル既定"

    try:
        # まずは希望どおりの processor_kwargs（既定では max_pixels 指定）でロードを試みる。
        model, used = _load(processor_kwargs)
    except (ImportError, ValueError, RuntimeError) as exc:
        # 非 Qwen の小型 student など processor_kwargs（max_pixels 等）非対応のモデル向けの救済。
        # Qwen 正常系は最初の試行で成功するためここには到達しない。
        if not processor_kwargs:
            raise
        logger.warning(
            "  processor_kwargs=%s が使えません（%s）。これを外して再試行します", processor_kwargs, exc
        )
        model, used = _load({})

    logger.info("  attn_implementation: %s", used)
    return model
