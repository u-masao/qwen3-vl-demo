"""合成キャプション付き画像データセットの生成。

prompts.py が作ったキャプション 1 文ごとに、対応する画像を 1 枚レンダリングする:

  * 通常（GPU）  → diffusers の text-to-image モデルで実生成。``image_gen.model_id`` で
                   SD-Turbo（既定）/ FLUX.2-klein（configs/flux.yaml）などを切り替えられる
  * ``smoke``    → 安価な合成スタブ画像（モデルのダウンロード不要・CPU 可）

結果は ``datasets.Dataset`` の 2 スプリットとして ``<data_dir>/train`` と
``<data_dir>/eval`` に保存する。各行のカラムは以下:

  * ``anchor``   (str)   キャプション ＝ 検索クエリ文
  * ``positive`` (Image) レンダリングした画像 ＝ 検索のターゲット
  * ``category`` (str)   被写体カテゴリ（緩い評価のオプションで使用）

カラム名の ``anchor`` / ``positive`` は Sentence Transformers の対照学習
（MultipleNegativesRankingLoss）が期待する慣習的な名前に合わせてある。詳しくは
train.py を参照。
"""

from __future__ import annotations

import argparse
import hashlib
import logging

from datasets import Dataset, Features, Value
from datasets import Image as HFImage
from PIL import Image

from .config import Config, add_config_args, config_from_args, resolve_dtype
from .prompts import Sample, build_captions

logger = logging.getLogger(__name__)


def _stub_image(text: str, size: int) -> Image.Image:
    """キャプションから決定的に決まる、依存ライブラリ不要のダミー画像を作る。

    スモークプロファイルで使用。拡散モデルをダウンロードせずに CPU だけで
    パイプラインを流すための代替。色はキャプションのハッシュから決めるので、
    異なるキャプションは異なる（しかし毎回同じ）色の画像になる。これにより
    「キャプションと画像が 1 対 1 で対応している」という前提だけは満たせる。
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    color = (digest[0], digest[1], digest[2])  # 先頭 3 バイトを RGB に流用
    return Image.new("RGB", (size, size), color)


def _generate_stub(samples: list[Sample], size: int) -> list[Image.Image]:
    """スタブ画像をサンプル数ぶんまとめて生成する。"""
    return [_stub_image(s.text, size) for s in samples]


def _generate_with_diffusers(samples: list[Sample], cfg: Config) -> list[Image.Image]:
    """diffusers の text-to-image モデルで画像をレンダリングする（モデル非依存）。

    ``AutoPipelineForText2Image`` がリポジトリの種類を自動判別するため、SD-Turbo でも
    FLUX.2-klein でも同じコードで動く。モデルごとの違い（ステップ数・guidance）は
    すべて設定（``image_gen.num_inference_steps`` / ``guidance_scale``）で吸収する:

      * ``stabilityai/sd-turbo``                  → steps=1, guidance=0.0
      * ``black-forest-labs/FLUX.2-klein-4b-fp8`` → steps=4, guidance=1.0（configs/flux.yaml）

    diffusers / torch はここで遅延 import する。これによりスモークモード
    （スタブ画像）では GPU 系の重い依存を一切ロードしない。
    """
    import time

    import torch
    from diffusers import AutoPipelineForText2Image

    dtype = resolve_dtype(cfg.dtype)
    logger.info("モデルロード開始: %s (dtype=%s)", cfg.image_gen.model_id, cfg.dtype)
    t0 = time.monotonic()
    pipe = AutoPipelineForText2Image.from_pretrained(
        cfg.image_gen.model_id,
        torch_dtype=dtype,
    )
    pipe = pipe.to(cfg.device)
    pipe.set_progress_bar_config(disable=True)
    logger.info("モデルロード完了: %.1f 秒", time.monotonic() - t0)

    images: list[Image.Image] = []
    bs = max(1, cfg.image_gen.batch_size)
    total = len(samples)
    # seed 固定の Generator で再現性を確保（同じ設定なら同じ画像が出る）。
    generator = torch.Generator(device=cfg.device).manual_seed(cfg.seed)
    t_batch_start = time.monotonic()
    for start in range(0, total, bs):
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
        done = min(start + bs, total)
        elapsed = time.monotonic() - t_batch_start
        logger.info("  生成 %d/%d 枚 (%.1f 秒, %.2f 枚/秒)", done, total, elapsed, done / elapsed)
    return images


def _build_split(samples: list[Sample], images: list[Image.Image]) -> Dataset:
    """サンプル（キャプション）と画像から 1 スプリット分の Dataset を組み立てる。

    ``positive`` カラムを ``datasets.Image`` 型にしておくことで、ディスク保存時に
    画像が適切にシリアライズされ、読み込み時には PIL 画像として自動デコードされる。
    """
    features = Features(
        {
            "anchor": Value("string"),
            "positive": HFImage(),
            "category": Value("string"),
            "subject": Value("string"),
        }
    )
    data = {
        "anchor": [s.text for s in samples],
        "positive": images,
        "category": [s.category for s in samples],
        "subject": [s.subject for s in samples],
    }
    return Dataset.from_dict(data, features=features)


def generate_dataset(cfg: Config) -> None:
    """設定に従って train / eval データセットを生成し、ディスクへ保存する。"""
    # train と eval で seed をずらし、キャプションが重複しないようにする。
    train_samples = build_captions(cfg.data.num_train, seed=cfg.seed)
    eval_samples = build_captions(cfg.data.num_eval, seed=cfg.seed + 10_000)

    # スモークプロファイル、または明示的に model_id="stub" の場合はスタブ画像を使う。
    use_stub = cfg.is_smoke or cfg.image_gen.model_id == "stub"

    logger.info("train %d 件 + eval %d 件の画像を生成します", cfg.data.num_train, cfg.data.num_eval)
    logger.info("  画像ソース: %s", "スタブ（合成画像）" if use_stub else cfg.image_gen.model_id)

    if use_stub:
        train_images = _generate_stub(train_samples, cfg.data.image_size)
        eval_images = _generate_stub(eval_samples, cfg.data.image_size)
    else:
        logger.info("train スプリット:")
        train_images = _generate_with_diffusers(train_samples, cfg)
        logger.info("eval スプリット:")
        eval_images = _generate_with_diffusers(eval_samples, cfg)

    train_ds = _build_split(train_samples, train_images)
    eval_ds = _build_split(eval_samples, eval_images)

    cfg.data_path.mkdir(parents=True, exist_ok=True)
    train_ds.save_to_disk(str(cfg.data_path / "train"))
    eval_ds.save_to_disk(str(cfg.data_path / "eval"))
    logger.info(
        "データセットを %s に保存しました（train=%d, eval=%d）",
        cfg.data_path,
        len(train_ds),
        len(eval_ds),
    )


def main() -> None:
    """CLI エントリポイント: ``python -m qwen3vl_demo.generate_data``。"""
    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser(description="合成キャプション付き画像データセットを生成する。")
    add_config_args(parser)
    args = parser.parse_args()
    cfg = config_from_args(args)
    generate_dataset(cfg)


if __name__ == "__main__":
    main()
