"""generate_data の純 Python 部（スタブ画像・Dataset 組み立て）のテスト。

torch / diffusers は不要（遅延 import のため）。datasets と Pillow のみ使用。
"""

from __future__ import annotations

from qwen3vl_demo.generate_data import _build_split, _stub_image
from qwen3vl_demo.prompts import build_captions


def test_stub_image_shape_and_mode():
    img = _stub_image("a cat", size=32)
    assert img.size == (32, 32)
    assert img.mode == "RGB"


def test_stub_image_deterministic():
    a = _stub_image("a fluffy cat", size=16)
    b = _stub_image("a fluffy cat", size=16)
    assert a.tobytes() == b.tobytes()


def test_stub_image_differs_by_text():
    a = _stub_image("a cat", size=16)
    b = _stub_image("a dog", size=16)
    # 異なるキャプションは異なる色になるはず（ハッシュ由来）。
    assert a.getpixel((0, 0)) != b.getpixel((0, 0))


def test_build_split_columns_and_length():
    samples = build_captions(5, seed=0)
    images = [_stub_image(s.text, 16) for s in samples]
    ds = _build_split(samples, images)
    assert len(ds) == 5
    assert set(ds.column_names) == {"anchor", "positive", "category"}
    row = ds[0]
    assert isinstance(row["anchor"], str)
    assert row["category"] in {s.category for s in samples}
