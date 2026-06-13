"""合成キャプション付き画像データセットの生成。

prompts.py が作ったキャプション 1 文ごとに、対応する画像を 1 枚レンダリングする:

  * 通常（GPU）  → diffusers の text-to-image モデルで実生成。``image_gen.model_id`` で
                   FLUX.2-klein-4B（既定）/ fp8 版（params_flux.yaml）などを切り替えられる
  * ``smoke``    → 安価な合成スタブ画像（モデルのダウンロード不要・CPU 可）

結果は ``datasets.Dataset`` の 2 スプリットとして ``<data_dir>/train`` と
``<data_dir>/eval`` に保存する。各行のカラムは以下:

  * ``anchor``   (str)   キャプション ＝ 画像生成プロンプト（学習・評価のクエリには使わない）
  * ``positive`` (Image) レンダリングした画像 ＝ 検索のターゲット
  * ``category`` (str)   被写体カテゴリ（緩い評価のオプションで使用）
  * ``subject``  (str)   被写体単語（"cat" など）
  * ``persona``  (str)   ペルソナ名（"user_alpha" など）＝ 嗜好ベース検索のクエリ／ラベル

カラム名の ``anchor`` / ``positive`` は Sentence Transformers の対照学習
（MultipleNegativesRankingLoss）が期待する慣習的な名前に合わせてある。ただし学習時は
train.py が ``persona`` 列を ``anchor`` に昇格させ、(ペルソナ名, 画像) のペアで学習する。
詳しくは train.py を参照。
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import logging
from collections.abc import Callable, Iterable, Iterator

from datasets import Dataset, Features, Value
from datasets import Image as HFImage
from PIL import Image

from .config import (
    Config,
    add_common_args,
    add_config_args,
    add_data_args,
    add_image_gen_args,
    config_from_args,
    resolve_dtype,
)
from .image_cache import ImageCache, derive_seed
from .prompts import Sample, build_captions
from .tracking import (
    DATA_EXPERIMENT_NAME,
    Timer,
    args_to_params,
    config_to_params,
    enable_system_metrics,
    log_metrics,
    start_run,
)

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


# プロンプト群とそれぞれの per-image シードを受け取り、画像を 1 枚ずつ順に yield するレンダラ。
# 1 枚ずつ返すことで、呼び出し側（_render_with_cache）が生成直後に逐次キャッシュへ書ける
# （スプリット全生成を待たずに保存＝途中終了でも進捗が残る）。diffusers 実装を差し替え可能に
# してキャッシュのオーケストレーションを GPU なしでテストできる。
GenerateFn = Callable[[list[str], list[int]], Iterable[Image.Image]]


def _render_with_cache(
    samples: list[Sample],
    cfg: Config,
    cache: ImageCache,
    generate_fn: GenerateFn,
    stats: dict[str, int] | None = None,
) -> list[Image.Image]:
    """キャッシュを参照しつつ画像をそろえる。ミス分だけ ``generate_fn`` で生成して書き戻す。

    キャッシュキーは「出力画像を一意に決める入力すべて」（モデル・プロンプト・seed・steps・
    guidance・サイズ・dtype）から作る。全件ヒットなら ``generate_fn`` は一度も呼ばれないので、
    呼び出し側はモデルロードを丸ごと省略できる（最大の利得）。

    ``generate_fn`` は画像を 1 枚ずつ yield するので、生成直後に ``cache.put`` する。これにより
    スプリットの全生成完了を待たずに保存され、途中で中断しても生成済みぶんはキャッシュに残る。

    ``stats`` を渡すと ``{"total", "hits", "misses"}`` を書き込む（MLflow 記録用。Issue #9）。
    """
    keys = [
        cache.key(
            model_id=cfg.image_gen.model_id,
            prompt=s.text,
            seed=cfg.seed,
            steps=cfg.image_gen.num_inference_steps,
            guidance=cfg.image_gen.guidance_scale,
            size=cfg.data.image_size,
            dtype=cfg.dtype,
        )
        for s in samples
    ]

    images: list[Image.Image | None] = [None] * len(samples)
    miss_idx: list[int] = []
    for i, key in enumerate(keys):
        cached = cache.get(key)
        if cached is None:
            miss_idx.append(i)
        else:
            images[i] = cached

    logger.info(
        "  キャッシュ: ヒット %d / 全 %d 枚（生成対象 %d 枚）",
        len(samples) - len(miss_idx),
        len(samples),
        len(miss_idx),
    )
    if stats is not None:
        stats.update(
            {"total": len(samples), "hits": len(samples) - len(miss_idx), "misses": len(miss_idx)}
        )

    if miss_idx:
        prompts = [samples[i].text for i in miss_idx]
        # 各画像はプロンプト単位の決定的シードで生成する（生成順・バッチ非依存）。
        seeds = [derive_seed(cfg.seed, p) for p in prompts]
        # generate_fn は順番に 1 枚ずつ yield する。生成直後にキャッシュへ書き込む（逐次保存）。
        produced = 0
        for local_i, img in enumerate(generate_fn(prompts, seeds)):
            gi = miss_idx[local_i]
            images[gi] = img
            cache.put(keys[gi], img)
            produced += 1
        if produced != len(miss_idx):
            raise RuntimeError(
                f"生成枚数がミス件数と一致しません（生成 {produced} / ミス {len(miss_idx)}）。"
            )

    return [img for img in images if img is not None]


def _make_diffusers_generator(cfg: Config) -> tuple[GenerateFn, Callable[[], None]]:
    """diffusers パイプラインを **一度だけ** ロードして共有する ``(generate_fn, close)`` を返す。

    ``AutoPipelineForText2Image`` がリポジトリの種類を自動判別するため、FLUX.2-klein でも
    他の diffusers モデルでも同じコードで動く。モデルごとの違い（ステップ数・guidance）は
    すべて設定（``image_gen.num_inference_steps`` / ``guidance_scale``）で吸収する:

      * ``black-forest-labs/FLUX.2-klein-4B``     → steps=4, guidance=1.0（既定 / params_default.yaml）
      * ``black-forest-labs/FLUX.2-klein-4b-fp8`` → steps=4, guidance=1.0（VRAM 節約版 / params_flux.yaml）

    パイプラインは最初のミス時に一度だけ遅延ロードし、以降の ``generate_fn`` 呼び出しで使い回す。
    train / eval を別々にロードすると FLUX が VRAM に二重に乗って枯渇し、生成が極端に遅くなる
    （16GB カードで実測 0.33→0.01 枚/秒）。そのため両スプリットで **同じパイプライン** を共有する。
    使い終わったら ``close()`` で破棄し VRAM を解放する。

    diffusers / torch はここで遅延 import する。これによりスモークモード（スタブ画像）や
    キャッシュ全ヒット時には GPU 系の重い依存を一切ロードしない。
    """
    import time

    import torch
    from diffusers import AutoPipelineForText2Image

    pipe_box: list = []

    def _load_pipe():
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
        return pipe

    def generate_fn(prompts: list[str], seeds: list[int]) -> Iterator[Image.Image]:
        if not pipe_box:
            pipe_box.append(_load_pipe())
        pipe = pipe_box[0]
        bs = max(1, cfg.image_gen.batch_size)
        total = len(prompts)
        t_batch_start = time.monotonic()
        done = 0
        for start in range(0, total, bs):
            batch_prompts = prompts[start : start + bs]
            # プロンプトごとの Generator を渡し、画像を生成順・バッチ非依存で決定的にする。
            generators = [
                torch.Generator(device=cfg.device).manual_seed(s) for s in seeds[start : start + bs]
            ]
            out = pipe(
                prompt=batch_prompts,
                num_inference_steps=cfg.image_gen.num_inference_steps,
                guidance_scale=cfg.image_gen.guidance_scale,
                height=cfg.data.image_size,
                width=cfg.data.image_size,
                generator=generators,
            )
            # バッチ分を 1 枚ずつ yield する（呼び出し側が逐次キャッシュへ書ける）。
            yield from out.images
            done = min(start + bs, total)
            elapsed = time.monotonic() - t_batch_start
            logger.info(
                "  生成 %d/%d 枚 (%.1f 秒, %.2f 枚/秒)", done, total, elapsed, done / elapsed
            )

    def close() -> None:
        """パイプラインを破棄して VRAM を解放する（未ロードなら no-op）。"""
        if pipe_box:
            pipe_box.clear()
            if cfg.device.startswith("cuda"):
                torch.cuda.empty_cache()

    return generate_fn, close


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
            "persona": Value("string"),
        }
    )
    data = {
        "anchor": [s.text for s in samples],
        "positive": images,
        "category": [s.category for s in samples],
        "subject": [s.subject for s in samples],
        "persona": [s.persona for s in samples],
    }
    return Dataset.from_dict(data, features=features)


def generate_dataset(cfg: Config, cli_args: argparse.Namespace | None = None) -> None:
    """設定に従って train / eval データセットを生成し、ディスクへ保存する。

    ``cli_args`` を渡すと MLflow Experiment ``"generate_data"`` に run として記録する
    （起動引数・全設定・所要時間・キャッシュのヒット/ミス・System Metrics。Issue #9）。
    None の場合（テスト等）は記録しない。
    """
    # train と eval で seed をずらし、キャプションが重複しないようにする。
    train_samples = build_captions(cfg.data.num_train, seed=cfg.seed)
    eval_samples = build_captions(cfg.data.num_eval, seed=cfg.seed + 10_000)

    # スモークプロファイル、または明示的に model_id="stub" の場合はスタブ画像を使う。
    use_stub = cfg.is_smoke or cfg.image_gen.model_id == "stub"

    logger.info("train %d 件 + eval %d 件の画像を生成します", cfg.data.num_train, cfg.data.num_eval)
    logger.info("  画像ソース: %s", "スタブ（合成画像）" if use_stub else cfg.image_gen.model_id)

    # MLflow: Experiment "generate_data" に run として記録（System Metrics・所要時間・全設定）。
    run_ctx = contextlib.nullcontext()
    if cli_args is not None:
        enable_system_metrics()
        params = {**args_to_params(cli_args), **config_to_params(cfg)}
        run_ctx = start_run(
            run_name="generate_data",
            params=params,
            tags={
                "stage": "generate_data",
                "image_source": "stub" if use_stub else cfg.image_gen.model_id,
            },
            experiment=DATA_EXPERIMENT_NAME,
        )

    metrics: dict[str, float] = {"num_train": len(train_samples), "num_eval": len(eval_samples)}

    with run_ctx:
        if use_stub:
            with Timer() as t_all:
                train_images = _generate_stub(train_samples, cfg.data.image_size)
                eval_images = _generate_stub(eval_samples, cfg.data.image_size)
            metrics["time.generate_total_sec"] = t_all.elapsed
        else:
            # 生成画像キャッシュ（同一入力の再生成・モデルロードをスキップ）。
            cache = ImageCache(cfg.image_cache_path, enabled=cfg.image_gen.cache_enabled)
            logger.info("  画像キャッシュ: %s", cache.root if cache.enabled else "無効")
            # パイプラインは 1 度だけロードし、train / eval で共有する（VRAM 二重消費を避ける）。
            tr_stats: dict[str, int] = {}
            ev_stats: dict[str, int] = {}
            generate_fn, close = _make_diffusers_generator(cfg)
            try:
                logger.info("train スプリット:")
                with Timer() as t_train:
                    train_images = _render_with_cache(
                        train_samples, cfg, cache, generate_fn, stats=tr_stats
                    )
                logger.info("eval スプリット:")
                with Timer() as t_eval:
                    eval_images = _render_with_cache(
                        eval_samples, cfg, cache, generate_fn, stats=ev_stats
                    )
            finally:
                close()
            misses = tr_stats.get("misses", 0) + ev_stats.get("misses", 0)
            metrics.update(
                {
                    "time.generate_train_sec": t_train.elapsed,
                    "time.generate_eval_sec": t_eval.elapsed,
                    "time.generate_total_sec": t_train.elapsed + t_eval.elapsed,
                    "cache.hits": tr_stats.get("hits", 0) + ev_stats.get("hits", 0),
                    "cache.misses": misses,
                    "cache.generated": misses,  # ミス分だけ実際に生成した枚数
                }
            )

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
        log_metrics(metrics)


def main() -> None:
    """CLI エントリポイント: ``python -m qwen3vl_demo.generate_data``。"""
    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser(description="合成キャプション付き画像データセットを生成する。")
    add_config_args(parser)
    add_common_args(parser)
    add_data_args(parser)
    add_image_gen_args(parser)
    args = parser.parse_args()
    cfg = config_from_args(args)
    generate_dataset(cfg, cli_args=args)


if __name__ == "__main__":
    main()
