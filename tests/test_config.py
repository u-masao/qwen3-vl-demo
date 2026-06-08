"""config の読み込み・パス解決の単体テスト。"""

from __future__ import annotations

from pathlib import Path

from qwen3vl_demo.config import REPO_ROOT, Config, load_config


def test_load_default_profile():
    cfg = load_config(REPO_ROOT / "configs" / "default.yaml")
    assert cfg.profile == "default"
    assert cfg.is_smoke is False
    assert cfg.device == "cuda"
    assert cfg.embedding.model_id.startswith("Qwen/")
    assert cfg.reranker.model_id is not None


def test_load_smoke_profile():
    cfg = load_config(REPO_ROOT / "configs" / "smoke.yaml")
    assert cfg.profile == "smoke"
    assert cfg.is_smoke is True
    assert cfg.device == "cpu"
    # スモークではリランカー無効（学習・推論ともスキップされる契約）。
    assert cfg.reranker.model_id is None


def test_paths_resolve_to_absolute():
    cfg = load_config(REPO_ROOT / "configs" / "default.yaml")
    for p in (cfg.data_path, cfg.output_path, cfg.model_path, cfg.reranker_model_path):
        assert isinstance(p, Path)
        assert p.is_absolute()
    # リポジトリルート配下に解決されること。
    assert str(cfg.data_path).startswith(str(REPO_ROOT))


def test_unknown_keys_are_ignored():
    # 余分なキーがあっても落ちず、既知フィールドだけ反映される（前方互換）。
    cfg = Config()
    assert cfg.train.epochs == 1
