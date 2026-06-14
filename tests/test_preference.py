"""preference（裏設定＝潜在嗖好モデル）の単体テスト（純 Python・依存なし）。

中心は「交互作用項が本当に非加法的か」「gamma がそれを制御するか」「argmax ラベル
（assign_persona）が交互作用を尊重するか」の検証。これがこのタスクの主張
（リランカーの伸びしろ＝嗖好の交互作用）の土台になる。
"""

from __future__ import annotations

import random

import pytest

from qwen3vl_demo import preference as pref


def _attrs_pattern(n: int) -> list[int]:
    """長さ n の決定的な 0/1 パターン（軸数に依存しないテスト用）。"""
    return [i % 2 for i in range(n)]


def test_build_model_shapes_and_personas():
    m = pref.build_model()
    assert len(m.axes) == len(pref.AXES)
    assert len(m.axes) == 7
    assert m.personas() == list(pref.PERSONA_MIX.keys())
    assert len(m.personas()) == 7
    for p, theta in m.persona_pref.items():
        assert len(theta) == len(m.axes), p
    assert len(m.global_pref) == len(m.axes)


def test_appeal_is_deterministic():
    m = pref.build_model()
    a = _attrs_pattern(len(m.axes))
    assert pref.appeal(m, "user_alpha", a) == pref.appeal(m, "user_alpha", a)


def _mixed_second_difference(m, persona: str, i: int, j: int) -> float:
    """f(11) - f(10) - f(01) + f(00) を軸 (i, j) について計算する。

    加法（線形）な appeal ならこの離散二階混合差分は 0 になる。非ゼロ＝交互作用の存在。
    他の軸は 0 に固定するので、(i, j) 以外のペアの交互作用項は寄与しない。
    """

    def attrs(vi: int, vj: int) -> list[int]:
        v = [0] * len(m.axes)
        v[i] = vi
        v[j] = vj
        return v

    f = lambda vi, vj: pref.appeal(m, persona, attrs(vi, vj))  # noqa: E731
    return f(1, 1) - f(1, 0) - f(0, 1) + f(0, 0)


def test_interaction_is_nonadditive():
    # ノイズ・人気項を消すと、各交互作用ペアの混合二階差分はちょうど gamma*coef になるはず。
    g = 1.5
    m = pref.build_model(gamma=g, lam=0.0, sigma=0.0)
    seen_any = False
    for persona, tris in m.interactions.items():
        for tri in tris:
            i, j, coef = int(tri[0]), int(tri[1]), tri[2]
            mixed = _mixed_second_difference(m, persona, i, j)
            assert mixed == pytest.approx(g * coef), (persona, i, j, coef)
            assert abs(mixed) > 1e-9  # 非ゼロ＝線形では表現できない構造
            seen_any = True
    assert seen_any


def test_gamma_zero_is_additive():
    # gamma=0 なら交互作用が消え、appeal は加法的（混合二階差分が 0）。
    # ＝旧タスクの再現条件：リランカーの伸びしろがほぼ無い状態。
    m = pref.build_model(gamma=0.0, lam=0.0, sigma=0.0)
    for persona in m.personas():
        for tri in m.interactions.get(persona, []):
            i, j, _ = int(tri[0]), int(tri[1]), tri[2]
            assert _mixed_second_difference(m, persona, i, j) == pytest.approx(0.0)


def test_assign_persona_matches_bruteforce_argmax():
    # assign_persona は全ペルソナの appeal の argmax と一致する純関数。
    m = pref.build_model()
    for bits in range(1 << len(m.axes)):
        attrs = [(bits >> k) & 1 for k in range(len(m.axes))]
        expected = max(m.personas(), key=lambda p: pref.appeal(m, p, attrs))
        got = pref.assign_persona(m, attrs)
        assert got in m.personas()
        assert pref.appeal(m, got, attrs) == pytest.approx(pref.appeal(m, expected, attrs))


def test_assign_persona_respects_interaction():
    # user_epsilon は (warmth∧vintage) に +2.0 の強い交互作用を持つ（相反する型の混合を解消）。
    # ノイズ・人気を消すと、warm かつ vintage の候補は epsilon に強く引き寄せられる。
    m = pref.build_model(gamma=3.0, lam=0.0, sigma=0.0)
    attrs = [0] * len(m.axes)
    attrs[0] = 1  # warmth
    attrs[1] = 1  # era=vintage
    # epsilon の appeal が交互作用ぶん底上げされ、単体線形では負けるはずの相手に勝つ。
    eps_score = pref.appeal(m, "user_epsilon", attrs)
    assert eps_score == max(pref.appeal(m, p, attrs) for p in m.personas())
    assert pref.assign_persona(m, attrs) == "user_epsilon"


def test_relevance_score_in_unit_interval():
    m = pref.build_model()
    for persona in m.personas():
        for bits in range(1 << len(m.axes)):
            attrs = [(bits >> k) & 1 for k in range(len(m.axes))]
            s = pref.relevance_score(m, persona, attrs)
            assert 0.0 <= s <= 1.0


def test_graded_relevance_covers_corpus():
    m = pref.build_model()
    n = len(m.axes)
    corpus = [[1] * n, [0] * n, _attrs_pattern(n), [(1 - (i % 2)) for i in range(n)]]
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
    # user_alpha は warmth(θ=0.6)・era(θ=0.6) が正。多数サンプルすれば 1 が優勢になるはず。
    m = pref.build_model(sharpness=2.0)
    rng = random.Random(123)
    n = 3000
    sums = [0] * len(m.axes)
    for _ in range(n):
        for k, v in enumerate(pref.sample_item_attributes(m, "user_alpha", rng)):
            sums[k] += v
    assert sums[0] / n > 0.6  # warmth
    assert sums[1] / n > 0.6  # era (vintage)


def test_attributes_to_fragments_and_codec():
    m = pref.build_model()
    n = len(m.axes)
    attrs = [1, 0, 1] + [0] * (n - 3)  # warmth=1, era=0, ornament=1, 残りは 0
    frags = pref.attributes_to_fragments(m, attrs)
    assert len(frags) == n
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
    n = len(m.axes)
    for persona in m.personas():
        for attrs in ([1] * n, _attrs_pattern(n), [(1 - (i % 2)) for i in range(n)]):
            assert pref.appeal(m2, persona, attrs) == pytest.approx(pref.appeal(m, persona, attrs))
        # ラベル付けも一致する。
        assert pref.assign_persona(m2, _attrs_pattern(n)) == pref.assign_persona(
            m, _attrs_pattern(n)
        )
