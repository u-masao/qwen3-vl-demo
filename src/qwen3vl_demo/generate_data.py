"""合成キャプション付き画像データセットの生成。

prompts.py が作ったキャプション 1 文ごとに、対応する画像を 1 枚レンダリングする:

  * ``default`` プロファイル → Stable Diffusion Turbo（``stabilityai/sd-turbo``）で実生成
  * ``smoke``   プロファイル → 安価な合成スタブ画像（モデルのダウンロード不要・CPU 可）

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

from datasets import Dataset, Features, Value
from datasets import Image as HFImage
from PIL import Image

from .config import Config, add_config_args, config_from_args, resolve_dtype
from .prompts import Sample, build_captions


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


def _generate_sd_turbo(samples: list[Sample], cfg: Config) -> list[Image.Image]:
    """SD-Turbo で画像をレンダリングする。

    diffusers / torch はここで遅延 import する。これによりスモークモード
    （スタブ画像）では GPU 系の重い依存を一切ロードしない。
    """
    import torch
    from diffusers import AutoPipelineForText2Image

    dtype = resolve_dtype(cfg.dtype)
    pipe = AutoPipelineForText2Image.from_pretrained(
        cfg.image_gen.model_id,
        torch_dtype=dtype,
    )
    pipe = pipe.to(cfg.device)
    pipe.set_progress_bar_config(disable=True)  # tqdm の進捗バーは抑制（自前で print する）

    images: list[Image.Image] = []
    bs = max(1, cfg.image_gen.batch_size)
    # seed 固定の Generator で再現性を確保（同じ設定なら同じ画像が出る）。
    generator = torch.Generator(device=cfg.device).manual_seed(cfg.seed)
    for start in range(0, len(samples), bs):
        batch = samples[start : start + bs]
        prompts = [s.text for s in batch]
        out = pipe(
            prompt=prompts,
            num_inference_steps=cfg.image_gen.num_inference_steps,  # SD-Turbo は 1〜4
            guidance_scale=cfg.image_gen.guidance_scale,            # Turbo は 0.0
            height=cfg.data.image_size,
            width=cfg.data.image_size,
            generator=generator,
        )
        images.extend(out.images)
        print(f"  生成 {min(start + bs, len(samples))}/{len(samples)} 枚")
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
        }
    )
    data = {
        "anchor": [s.text for s in samples],
        "positive": images,
        "category": [s.category for s in samples],
    }
    return Dataset.from_dict(data, features=features)


def generate_dataset(cfg: Config) -> None:
    """設定に従って train / eval データセットを生成し、ディスクへ保存する。"""
    # train と eval で seed をずらし、キャプションが重複しないようにする。
    train_samples = build_captions(cfg.data.num_train, seed=cfg.seed)
    eval_samples = build_captions(cfg.data.num_eval, seed=cfg.seed + 10_000)

    # スモークプロファイル、または明示的に model_id="stub" の場合はスタブ画像を使う。
    use_stub = cfg.is_smoke or cfg.image_gen.model_id == "stub"

    print(f"train {cfg.data.num_train} 件 + eval {cfg.data.num_eval} 件の画像を生成します")
    print(f"  画像ソース: {'スタブ（合成画像）' if use_stub else cfg.image_gen.model_id}")

    if use_stub:
        train_images = _generate_stub(train_samples, cfg.data.image_size)
        eval_images = _generate_stub(eval_samples, cfg.data.image_size)
    else:
        print("train スプリット:")
        train_images = _generate_sd_turbo(train_samples, cfg)
        print("eval スプリット:")
        eval_images = _generate_sd_turbo(eval_samples, cfg)

    train_ds = _build_split(train_samples, train_images)
    eval_ds = _build_split(eval_samples, eval_images)

    cfg.data_path.mkdir(parents=True, exist_ok=True)
    train_ds.save_to_disk(str(cfg.data_path / "train"))
    eval_ds.save_to_disk(str(cfg.data_path / "eval"))
    print(f"データセットを {cfg.data_path} に保存しました（train={len(train_ds)}, eval={len(eval_ds)}）")


def main() -> None:
    """CLI エントリポイント: ``python -m qwen3vl_demo.generate_data``。"""
    parser = argparse.ArgumentParser(description="合成キャプション付き画像データセットを生成する。")
    add_config_args(parser)
    args = parser.parse_args()
    cfg = config_from_args(args)
    generate_dataset(cfg)


if __name__ == "__main__":
    main()
