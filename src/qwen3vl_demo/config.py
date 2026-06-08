"""Configuration loading: YAML -> nested dataclasses, with a small CLI helper.

Every entry-point module accepts ``--config PATH`` (defaulting to
``configs/default.yaml``) and ``--profile {default,smoke}`` as a convenience
shortcut that maps to ``configs/<profile>.yaml``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Repo root = three levels up from this file (src/qwen3vl_demo/config.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


@dataclass
class Paths:
    data_dir: str = "data"
    output_dir: str = "outputs"
    model_dir: str = "outputs/model"


@dataclass
class DataCfg:
    num_train: int = 200
    num_eval: int = 50
    image_size: int = 512
    relevant_same_category: bool = False


@dataclass
class ImageGenCfg:
    model_id: str = "stabilityai/sd-turbo"
    num_inference_steps: int = 1
    guidance_scale: float = 0.0
    batch_size: int = 8


@dataclass
class EmbeddingCfg:
    model_id: str = "Qwen/Qwen3-VL-Embedding-2B"
    attn_implementation: str = "flash_attention_2"
    max_pixels: int | None = None
    query_prompt_name: str | None = None


@dataclass
class RerankerCfg:
    model_id: str | None = "Qwen/Qwen3-VL-Reranker-2B"
    top_k: int = 10


@dataclass
class TrainCfg:
    epochs: int = 1
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2.0e-5
    warmup_ratio: float = 0.1
    gradient_checkpointing: bool = True
    eval_steps: int = 50
    save_steps: int = 50
    logging_steps: int = 10


@dataclass
class Config:
    profile: str = "default"
    seed: int = 42
    device: str = "cuda"
    dtype: str = "bfloat16"
    paths: Paths = field(default_factory=Paths)
    data: DataCfg = field(default_factory=DataCfg)
    image_gen: ImageGenCfg = field(default_factory=ImageGenCfg)
    embedding: EmbeddingCfg = field(default_factory=EmbeddingCfg)
    reranker: RerankerCfg = field(default_factory=RerankerCfg)
    train: TrainCfg = field(default_factory=TrainCfg)

    # --- convenience accessors as absolute paths ------------------------------
    def _abs(self, p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else REPO_ROOT / path

    @property
    def data_path(self) -> Path:
        return self._abs(self.paths.data_dir)

    @property
    def output_path(self) -> Path:
        return self._abs(self.paths.output_dir)

    @property
    def model_path(self) -> Path:
        return self._abs(self.paths.model_dir)

    @property
    def is_smoke(self) -> bool:
        return self.profile == "smoke"


def _build(section_cls, data: dict[str, Any] | None):
    """Instantiate a dataclass from a dict, ignoring unknown keys."""
    data = data or {}
    known = {f.name for f in section_cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    return section_cls(**{k: v for k, v in data.items() if k in known})


def load_config(path: str | Path) -> Config:
    """Load a YAML config file into a :class:`Config`."""
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

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
    """Add the shared ``--config`` / ``--profile`` arguments to a parser."""
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML config file. Overrides --profile when set.",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="default",
        choices=["default", "smoke"],
        help="Named config under configs/<profile>.yaml (default: default).",
    )


def config_from_args(args: argparse.Namespace) -> Config:
    """Resolve a :class:`Config` from parsed args (--config wins over --profile)."""
    if args.config:
        path = Path(args.config)
    else:
        path = CONFIG_DIR / f"{args.profile}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return load_config(path)


def resolve_dtype(name: str):
    """Map a dtype string to a torch dtype (imported lazily)."""
    import torch

    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]
