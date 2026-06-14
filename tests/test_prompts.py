"""prompts.build_captions の単体テスト（純 Python・依存なし）。"""

from __future__ import annotations

import pytest

from qwen3vl_demo import preference as pref
from qwen3vl_demo.prompts import SUBJECTS, build_captions, build_captions_preference


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


# --- preference タスク（並列の生成バリアント）-------------------------------
_ALL_SUBJECTS = {subj for subs in SUBJECTS.values() for subj in subs}


def test_preference_count_and_schema():
    model = pref.build_model()
    samples = build_captions_preference(30, seed=1, model=model)
    assert len(samples) == 30
    # subject タスクと同一スキーマ（下流が変更不要であることの担保）。
    for s in samples:
        assert isinstance(s.text, str) and s.text
        assert s.subject in _ALL_SUBJECTS
        assert s.category in SUBJECTS
        assert s.persona in model.personas()


def test_preference_deterministic_for_same_seed():
    model = pref.build_model()
    a = build_captions_preference(20, seed=42, model=model)
    b = build_captions_preference(20, seed=42, model=model)
    assert [(s.text, s.persona) for s in a] == [(s.text, s.persona) for s in b]


def test_preference_text_embeds_subject():
    model = pref.build_model()
    samples = build_captions_preference(20, seed=3, model=model)
    for s in samples:
        assert s.text.startswith("a photo of a ")
        assert s.subject in s.text


def test_preference_uniqueness():
    model = pref.build_model()
    samples = build_captions_preference(50, seed=7, model=model)
    texts = [s.text for s in samples]
    assert len(set(texts)) == len(texts)
