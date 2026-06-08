"""Helpers for loading the Sentence Transformers embedding model.

Centralised so ``train.py`` and ``evaluate.py`` build the model identically.
"""

from __future__ import annotations

from .config import Config, resolve_dtype


def load_embedding_model(cfg: Config, model_id: str | None = None):
    """Load a :class:`SentenceTransformer` embedding model per the config.

    ``model_id`` overrides ``cfg.embedding.model_id`` (used to load the
    fine-tuned model from disk for evaluation). Falls back from
    ``flash_attention_2`` to ``sdpa`` if flash-attn isn't installed.
    """
    from sentence_transformers import SentenceTransformer

    model_id = model_id or cfg.embedding.model_id

    model_kwargs: dict = {}
    processor_kwargs: dict = {}

    # dtype only matters on GPU; keep float32 on CPU/smoke for portability.
    if cfg.device != "cpu":
        model_kwargs["torch_dtype"] = resolve_dtype(cfg.dtype)
    if cfg.embedding.attn_implementation:
        model_kwargs["attn_implementation"] = cfg.embedding.attn_implementation
    if cfg.embedding.max_pixels:
        # Cap the number of image patches to bound VRAM for big images.
        processor_kwargs["max_pixels"] = cfg.embedding.max_pixels

    def _build(mk: dict, pk: dict):
        kwargs: dict = {"device": cfg.device}
        if mk:
            kwargs["model_kwargs"] = mk
        if pk:
            kwargs["processor_kwargs"] = pk
        return SentenceTransformer(model_id, **kwargs)

    try:
        model = _build(model_kwargs, processor_kwargs)
    except (ImportError, ValueError, RuntimeError) as exc:
        # flash-attn missing or unsupported processor kwargs -> retry with sdpa.
        if model_kwargs.get("attn_implementation") == "flash_attention_2":
            print(f"  flash_attention_2 unavailable ({exc}); retrying with sdpa")
            model_kwargs["attn_implementation"] = "sdpa"
            try:
                model = _build(model_kwargs, processor_kwargs)
            except (ImportError, ValueError, RuntimeError) as exc2:
                print(f"  sdpa attention also failed ({exc2}); using model defaults")
                model_kwargs.pop("attn_implementation", None)
                model = _build(model_kwargs, processor_kwargs)
        else:
            raise

    return model
