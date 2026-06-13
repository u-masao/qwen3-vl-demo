"""tracking ヘルパの単体テスト（純 Python・mlflow を実起動しない）。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

from qwen3vl_demo.config import _UNSET
from qwen3vl_demo.tracking import _sanitize_key, args_to_params, config_to_params


def test_args_to_params_prefixes_and_filters_unset():
    args = argparse.Namespace(seed=42, profile="default", model=None, max_pixels=_UNSET)
    params = args_to_params(args)
    # 渡した引数は args.* プレフィックス付きで残る（None も保持）。
    assert params == {"args.seed": 42, "args.profile": "default", "args.model": None}
    # 未指定オーバーライド（_UNSET センチネル）は除外される。
    assert "args.max_pixels" not in params


def test_sanitize_key_replaces_at():
    # MLflow のキーで不許可な '@' は '_at_' に置換する。
    assert _sanitize_key("ndcg@10") == "ndcg_at_10"
    assert _sanitize_key("recall@1") == "recall_at_1"


def test_sanitize_key_keeps_allowed_chars():
    # 英数・_ - . / : 空白は許可されるのでそのまま。
    key = "synthetic-image-retrieval_cosine_ndcg@10"
    assert _sanitize_key(key) == "synthetic-image-retrieval_cosine_ndcg_at_10"


def test_config_to_params_flattens_nested_dataclass():
    @dataclass
    class Inner:
        model_id: str = "m"
        max_pixels: int | None = None

    @dataclass
    class Cfg:
        seed: int = 42
        embedding: Inner = field(default_factory=Inner)

    params = config_to_params(Cfg())
    # ネストした dataclass は cfg.<section>.<key> のドット区切りに平坦化される。
    assert params["cfg.seed"] == 42
    assert params["cfg.embedding.model_id"] == "m"
    assert params["cfg.embedding.max_pixels"] is None


def test_config_to_params_non_dataclass_returns_empty():
    assert config_to_params({"seed": 1}) == {}
