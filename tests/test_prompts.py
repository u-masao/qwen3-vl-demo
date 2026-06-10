"""prompts.build_captions の単体テスト（純 Python・依存なし）。"""

from __future__ import annotations

import pytest

from qwen3vl_demo.prompts import SUBJECTS, build_captions


def test_count_and_type():
    samples = build_captions(20, seed=1)
    assert len(samples) == 20
    assert all(isinstance(s.text, str) and s.text for s in samples)


def test_deterministic_for_same_seed():
    a = build_captions(15, seed=42)
    b = build_captions(15, seed=42)
    assert [s.text for s in a] == [s.text for s in b]


def test_different_seed_differs():
    a = build_captions(15, seed=1)
    b = build_captions(15, seed=2)
    # まったく同一になる確率は無視できる。少なくとも 1 件は違うはず。
    assert [s.text for s in a] != [s.text for s in b]


def test_uniqueness():
    samples = build_captions(50, seed=7)
    texts = [s.text for s in samples]
    assert len(set(texts)) == len(texts)


def test_categories_are_valid():
    samples = build_captions(30, seed=3)
    assert all(s.category in SUBJECTS for s in samples)


def test_raises_when_too_many_requested():
    # 少ない試行回数では要求件数の一意なキャプションを作れない -> ValueError。
    # max_attempts を小さく渡すことで、巨大なループを回さず即座に検証する。
    with pytest.raises(ValueError):
        build_captions(100, seed=0, max_attempts=10)
