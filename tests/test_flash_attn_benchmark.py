"""flash_attention_2 vs eager のスループット比較ベンチマーク。

CUDA GPU と flash-attn パッケージが必要（uv sync --extra gpu）。
Qwen3-VL-Embedding-2B を bf16 でロードし、448x448 画像バッチの encode 速度を比較する。

【なぜ sdpa ではなく eager と比べるのか】
PyTorch 2.x の sdpa (scaled_dot_product_attention) は Ada 世代の GPU で
内部的に flash-attention 相当のカーネルを使うため、flash_attention_2 との差が
ほとんど出ない。一方 eager は O(n²) の素実装なので差が明確になる。
448x448 画像 (~1024 パッチ/枚) では flash が eager の約 1.4× 高速になる。
"""

from __future__ import annotations

import time

import pytest

try:
    import flash_attn  # noqa: F401

    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

try:
    import torch

    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False

pytestmark = [
    pytest.mark.skipif(not HAS_CUDA, reason="CUDA GPU が必要"),
    pytest.mark.skipif(
        not HAS_FLASH_ATTN, reason="flash-attn が未インストール（uv sync --extra gpu）"
    ),
]

MODEL_ID = "Qwen/Qwen3-VL-Embedding-2B"
IMAGE_SIZE = 448  # 448x448 → ~1024 パッチ → flash attention の優位性が出やすい系列長
BATCH_SIZE = 4
WARMUP_RUNS = 2
BENCH_RUNS = 5
# flash_attention_2 が eager より何倍以上速ければ OK か（観測値 ~1.4× の安全マージン）
SPEEDUP_THRESHOLD = 1.20


def _make_images(n: int, size: int):
    """n 枚の合成 RGB 画像 (PIL) を返す。"""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(42)
    return [
        Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8)) for _ in range(n)
    ]


def _bench(attn_impl: str, images: list) -> float:
    """指定 attn_implementation でモデルをロードし、encode の平均時間（秒）を返す。"""
    from qwen3vl_demo.config import Config
    from qwen3vl_demo.models import load_embedding_model

    cfg = Config(device="cuda", dtype="bfloat16")
    cfg.embedding.model_id = MODEL_ID
    cfg.embedding.attn_implementation = attn_impl
    cfg.embedding.max_pixels = None

    model = load_embedding_model(cfg)

    for _ in range(WARMUP_RUNS):
        model.encode(images, batch_size=BATCH_SIZE, convert_to_tensor=True)
        torch.cuda.synchronize()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(BENCH_RUNS):
        model.encode(images, batch_size=BATCH_SIZE, convert_to_tensor=True)
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / BENCH_RUNS

    del model
    torch.cuda.empty_cache()
    return elapsed


def test_flash_attention_faster_than_eager():
    """flash_attention_2 の encode が eager より SPEEDUP_THRESHOLD 倍以上高速であること。

    eager は O(n²) の素実装。448x448 画像 (~1024 パッチ) では
    flash_attention_2 が約 1.4× 高速になることを確認する。
    """
    images = _make_images(BATCH_SIZE, IMAGE_SIZE)

    t_flash = _bench("flash_attention_2", images)
    t_eager = _bench("eager", images)
    t_sdpa = _bench("sdpa", images)  # 参考値: sdpa は内部で同等カーネルを使うため flash と同程度

    speedup = t_eager / t_flash
    print(f"\nflash_attention_2 : {t_flash * 1000:.1f} ms/batch")
    print(f"eager             : {t_eager * 1000:.1f} ms/batch")
    print(f"sdpa (参考)        : {t_sdpa * 1000:.1f} ms/batch")
    print(f"speedup (eager/flash): {speedup:.2f}x  (閾値: {SPEEDUP_THRESHOLD}x)")

    assert speedup >= SPEEDUP_THRESHOLD, (
        f"flash_attention_2 ({t_flash * 1000:.1f} ms) の eager ({t_eager * 1000:.1f} ms) "
        f"に対する速度比が {speedup:.2f}x で閾値 {SPEEDUP_THRESHOLD}x を下回った"
    )
