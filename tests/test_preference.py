"""preference（裏設定＝潜在嗖好モデル）の単体テスト（純 Python・依存なし）。

中心は「交互作用項が本当に非加法的か」「gamma がそれを制御するか」の検証。
これがこのタスクの主張（リランカーの伸びしろ＝嗖好の交互作用）の土台になる。
"""

from __future__ import annotations

import random

import pytest

from qwen3vl_demo import preference as pref


def test_build_model_shapes_and_personas():
    m = pref.build_model()
    assert len(m.axes) == len(pref.AXES)
    assert m.personas() == list(pref.PERSONA_MIX.keys())
    assert len(m.personas()) == 7
    for p, theta in m.persona_pref.items():
        assert len(theta) == len(m.axes), p
    assert len(m.global_pref) == len(m.axes)


def test_appeal_is_deterministic():
    m = pref.build_model()
    a = [1, 0, 1, 0]
    assert pref.appeal(m, "user_alpha", a) == pref.appeal(m, "user_alpha", a)


def _mixed_second_difference(m, persona: str, i: int, j: int) -> float:
    """f(11) - f(10) - f(01) + f(00) を軸 (i, j) について計算する。

    加法（線形）な appeal ならこの離散二階混合差分は 0 になる。非ゼロ＝交互作用の存在。
    """

    def attrs(vi: int, vj: int) -> list[int]:
        v = [0] * len(m.axes)
        v[i] = vi
        v[j] = vj
        return v

    f = lambda vi, vj: pref.appeal(m, persona, attrs(vi, vj))  # noqa: E731
    return f(1, 1) - f(1, 0) - f(0, 1) + f(0, 0)


def test_interaction_is_nonadditive():
    # user_alpha は (warmth=0, ornament=2) に coef=-1 の交互作用を持つ。
    # ノイズ・人気項を消すと、混合二階差分はちょうど gamma*coef になるはず。
    g = 1.5
    m = pref.build_model(gamma=g, lam=0.0, sigma=0.0)
    i, j, coef = 0, 2, -1.0
    mixed = _mixed_second_difference(m, "user_alpha", i, j)
    assert mixed == pytest.approx(g * coef)
    # 非ゼロ＝線形（加法）モデルでは表現できない構造であることの確認。
    assert abs(mixed) > 1e-9


def test_gamma_zero_is_additive():
    # gamma=0 なら交互作用が消え、appeal は加法的（混合二階差分が 0）。
    # ＝旧タスクの再現条件：リランカーの伸びしろがほぼ無い状態。
    m = pref.build_model(gamma=0.0, lam=0.0, sigma=0.0)
    for persona in m.personas():
        for tri in m.interactions.get(persona, []):
            i, j, _ = int(tri[0]), int(tri[1]), tri[2]
            assert _mixed_second_difference(m, persona, i, j) == pytest.approx(0.0)


def test_relevance_score_in_unit_interval():
    m = pref.build_model()
    for persona in m.personas():
        for bits in range(1 << len(m.axes)):
            attrs = [(bits >> k) & 1 for k in range(len(m.axes))]
            s = pref.relevance_score(m, persona, attrs)
            assert 0.0 <= s <= 1.0


def test_graded_relevance_covers_corpus():
    m = pref.build_model()
    corpus = [[1, 0, 1, 0], [0, 0, 0, 0], [1, 1, 1, 1], [0, 1, 0, 1]]
    grades = pref.graded_relevance(m, "user_beta", corpus)
    assert set(grades.keys()) == set(range(len(corpus)))
    assert all(0.0 <= v <= 1.0 for v in grades.values())


def test_sample_attributes_deterministic_and_shaped():
    m = pref.build_model()
    a = pref.sample_item_attributes(m, "user_alpha", random.Random(0))
    b = pref.sample_item_attributes(m, "user_alpha", random.Random(0))
    assert a == b
    assert len(a) == len(m.axes)
    assert all(v in (0, 1) for v in a)


def test_sample_attributes_follow_preference():
    # user_alpha は全軸で θ>0（warmth/era が特に強い）。多数サンプルすれば 1 が優勢になるはず。
    m = pref.build_model(sharpness=2.0)
    rng = random.Random(123)
    n = 3000
    sums = [0] * len(m.axes)
    for _ in range(n):
        for k, v in enumerate(pref.sample_item_attributes(m, "user_alpha", rng)):
            sums[k] += v
    # warmth(θ=0.7), era(θ=0.7) は明確に 1 優勢。
    assert sums[0] / n > 0.6
    assert sums[1] / n > 0.6


def test_attributes_to_fragments_and_codec():
    m = pref.build_model()
    attrs = [1, 0, 1, 0]
    frags = pref.attributes_to_fragments(m, attrs)
    assert frags[0] == "warm-toned"  # warmth=1
    assert frags[1] == "modern"  # era=0
    assert frags[2] == "ornate, intricately detailed"  # ornament=1
    # encode/decode 往復
    assert pref.decode_attributes(pref.encode_attributes(attrs)) == attrs
    assert pref.decode_attributes("") == []


def test_save_load_roundtrip_preserves_appeal(tmp_path):
    m = pref.build_model(gamma=1.2, lam=0.4, sigma=0.05, seed=7)
    path = tmp_path / "preference_model.json"
    pref.save_model(m, path)
    m2 = pref.load_model(path)
    assert m2.personas() == m.personas()
    for persona in m.personas():
        for attrs in ([1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 1, 1]):
            assert pref.appeal(m2, persona, attrs) == pytest.approx(pref.appeal(m, persona, attrs))
