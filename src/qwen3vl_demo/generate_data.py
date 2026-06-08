"""Generate a synthetic captioned-image dataset.

For each template-generated caption we render one image:
  * ``default`` profile -> Stable Diffusion Turbo (``stabilityai/sd-turbo``)
  * ``smoke``   profile -> a cheap synthetic stub image (no model download)

The result is two ``datasets.Dataset`` splits saved under ``<data_dir>/train``
and ``<data_dir>/eval`` with columns:
  * ``anchor``   (str)   - the caption / retrieval query text
  * ``positive`` (Image) - the rendered image (the retrieval target)
  * ``category`` (str)   - subject category, for optional looser evaluation
"""

from __future__ import annotations

import argparse
import hashlib

from datasets import Dataset, Features, Value
from datasets import Image as HFImage
from PIL import Image

from .config import Config, add_config_args, config_from_args, resolve_dtype
from .prompts import Sample, build_captions


def _stub_image(text: str, size: int) -> Image.Image:
    """Deterministic, dependency-free placeholder image derived from the text.

    Used by the smoke profile so the pipeline runs on CPU without downloading
    a diffusion model. The colour is a hash of the caption, so different
    captions get visibly different (but stable) images.
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    color = (digest[0], digest[1], digest[2])
    return Image.new("RGB", (size, size), color)


def _generate_stub(samples: list[Sample], size: int) -> list[Image.Image]:
    return [_stub_image(s.text, size) for s in samples]


def _generate_sd_turbo(samples: list[Sample], cfg: Config) -> list[Image.Image]:
    """Render images with SD-Turbo. Imported lazily so smoke mode needs no GPU deps."""
    import torch
    from diffusers import AutoPipelineForText2Image

    dtype = resolve_dtype(cfg.dtype)
    pipe = AutoPipelineForText2Image.from_pretrained(
        cfg.image_gen.model_id,
        torch_dtype=dtype,
    )
    pipe = pipe.to(cfg.device)
    pipe.set_progress_bar_config(disable=True)

    images: list[Image.Image] = []
    bs = max(1, cfg.image_gen.batch_size)
    generator = torch.Generator(device=cfg.device).manual_seed(cfg.seed)
    for start in range(0, len(samples), bs):
        batch = samples[start : start + bs]
        prompts = [s.text for s in batch]
        out = pipe(
            prompt=prompts,
            num_inference_steps=cfg.image_gen.num_inference_steps,
            guidance_scale=cfg.image_gen.guidance_scale,
            height=cfg.data.image_size,
            width=cfg.data.image_size,
            generator=generator,
        )
        images.extend(out.images)
        print(f"  generated {min(start + bs, len(samples))}/{len(samples)} images")
    return images


def _build_split(samples: list[Sample], images: list[Image.Image]) -> Dataset:
    features = Features(
        {
            "anchor": Value("string"),
            "positive": HFImage(),
            "category": Value("string"),
        }
    )
    data = {
        "anchor": [s.text for s in samples],
        "positive": images,
        "category": [s.category for s in samples],
    }
    return Dataset.from_dict(data, features=features)


def generate_dataset(cfg: Config) -> None:
    # Use different seeds for train/eval so captions don't overlap.
    train_samples = build_captions(cfg.data.num_train, seed=cfg.seed)
    eval_samples = build_captions(cfg.data.num_eval, seed=cfg.seed + 10_000)

    use_stub = cfg.is_smoke or cfg.image_gen.model_id == "stub"

    print(f"Generating {cfg.data.num_train} train + {cfg.data.num_eval} eval images")
    print(f"  image source: {'stub (synthetic)' if use_stub else cfg.image_gen.model_id}")

    if use_stub:
        train_images = _generate_stub(train_samples, cfg.data.image_size)
        eval_images = _generate_stub(eval_samples, cfg.data.image_size)
    else:
        print("train split:")
        train_images = _generate_sd_turbo(train_samples, cfg)
        print("eval split:")
        eval_images = _generate_sd_turbo(eval_samples, cfg)

    train_ds = _build_split(train_samples, train_images)
    eval_ds = _build_split(eval_samples, eval_images)

    cfg.data_path.mkdir(parents=True, exist_ok=True)
    train_ds.save_to_disk(str(cfg.data_path / "train"))
    eval_ds.save_to_disk(str(cfg.data_path / "eval"))
    print(f"Saved dataset to {cfg.data_path} (train={len(train_ds)}, eval={len(eval_ds)})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic captioned-image dataset.")
    add_config_args(parser)
    args = parser.parse_args()
    cfg = config_from_args(args)
    generate_dataset(cfg)


if __name__ == "__main__":
    main()
