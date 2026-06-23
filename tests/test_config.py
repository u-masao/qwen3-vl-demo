"""config の読み込み・パス解決・CLI オーバーライドの単体テスト。"""

from __future__ import annotations

import argparse
from pathlib import Path

from qwen3vl_demo.config import (
    REPO_ROOT,
    Config,
    add_common_args,
    add_config_args,
    add_data_args,
    add_preference_args,
    add_train_args,
    config_from_args,
    load_config,
)


def test_load_default_profile():
    cfg = load_config(REPO_ROOT / "params_default.yaml")
    assert cfg.profile == "default"
    assert cfg.is_smoke is False
    assert cfg.device == "cuda"
    assert cfg.embedding.model_id.startswith("Qwen/")
    assert cfg.reranker.model_id is not None


def test_load_smoke_profile():
    cfg = load_config(REPO_ROOT / "params_smoke.yaml")
    assert cfg.profile == "smoke"
    assert cfg.is_smoke is True
    assert cfg.device == "cpu"
    # スモークではリランカー無効（学習・推論ともスキップされる契約）。
    assert cfg.reranker.model_id is None
    assert cfg.embedding.max_pixels is None


def test_load_flux_profile():
    cfg = load_config(REPO_ROOT / "params_flux.yaml")
    assert cfg.profile == "flux"
    assert cfg.is_smoke is False
    # FLUX.2-klein を使い、4 ステップ蒸留・guidance=1.0 が設定されていること。
    assert "FLUX.2-klein" in cfg.image_gen.model_id
    assert cfg.image_gen.num_inference_steps == 4
    assert cfg.image_gen.guidance_scale == 1.0


def test_active_params_yaml_loads():
    # 有効プロファイル（make use-* がコピーするファイル）が読めること。
    cfg = load_config(REPO_ROOT / "params.yaml")
    assert cfg.profile in ("default", "smoke", "flux")


def test_paths_resolve_to_absolute():
    cfg = load_config(REPO_ROOT / "params_default.yaml")
    for p in (cfg.data_path, cfg.output_path, cfg.model_path, cfg.reranker_model_path):
        assert isinstance(p, Path)
        assert p.is_absolute()
    # リポジトリルート配下に解決されること。
    assert str(cfg.data_path).startswith(str(REPO_ROOT))


def test_unknown_keys_are_ignored():
    # 余分なキーがあっても落ちず、既知フィールドだけ反映される（前方互換）。
    cfg = Config()
    assert cfg.train.epochs == 1


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_config_args(parser)
    add_common_args(parser)
    add_train_args(parser)
    return parser.parse_args(argv)


def test_cli_overrides_applied():
    # ベースは default プロファイル、--epochs / --lr / --seed / --data-dir を上書き。
    args = _parse(
        [
            "--profile",
            "default",
            "--epochs",
            "3",
            "--lr",
            "1e-4",
            "--seed",
            "7",
            "--data-dir",
            "tmp_data",
        ]
    )
    cfg = config_from_args(args)
    assert cfg.profile == "default"
    assert cfg.train.epochs == 3
    assert cfg.train.learning_rate == 1e-4
    assert cfg.seed == 7
    assert cfg.paths.data_dir == "tmp_data"
    # 指定していない値はベース YAML のまま。
    assert cfg.train.warmup_ratio == 0.1


def test_no_overrides_keep_base_values():
    # オーバーライド未指定なら params_default.yaml の値がそのまま使われる。
    args = _parse(["--profile", "default"])
    cfg = config_from_args(args)
    assert cfg.train.epochs == 1
    assert cfg.seed == 1  # default プロファイルの seed は 1（ce0404b で 42→1 に変更）


def test_bool_override_value_style():
    # gradient_checkpointing は値指定（DVC の ${...} 展開を受け取れる）。
    args = _parse(["--profile", "default", "--gradient-checkpointing", "false"])
    cfg = config_from_args(args)
    assert cfg.train.gradient_checkpointing is False


def _parse_data_pref(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_config_args(parser)
    add_common_args(parser)
    add_data_args(parser)
    add_preference_args(parser)
    return parser.parse_args(argv)


def test_default_task_is_preference():
    # 既定は preference（gamma=2.0）。subject は --task subject の legacy opt-in。
    cfg = load_config(REPO_ROOT / "params_default.yaml")
    assert cfg.data.task == "preference"
    assert cfg.preference.gamma == 2.0
    assert cfg.preference.lam == 0.3


def test_task_and_preference_overrides():
    # data.task と preference.* が CLI で上書きでき、未指定はベース YAML のままになる。
    args = _parse_data_pref(
        [
            "--profile",
            "default",
            "--task",
            "preference",
            "--pref-gamma",
            "3.5",
            "--pref-sigma",
            "0.0",
        ]
    )
    cfg = config_from_args(args)
    assert cfg.data.task == "preference"
    assert cfg.preference.gamma == 3.5
    assert cfg.preference.sigma == 0.0
    assert cfg.preference.lam == 0.3  # 未指定はベース値のまま
