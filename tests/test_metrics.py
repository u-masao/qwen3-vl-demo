"""metrics.py（per-persona マクロ集計・ばらつき）の単体テスト。GPU 不要。"""

from __future__ import annotations

import math

from qwen3vl_demo.metrics import (
    _bootstrap_ci,
    ir_metrics,
    macro_summary,
    per_persona_metrics,
)


def test_ir_metrics_basic():
    # 正解が 1 位 -> recall/ndcg/mrr=1。
    m = ir_metrics([[0, 1, 2]], [{0}], ks=[1, 3])
    assert m["recall@1"] == 1.0
    assert m["ndcg@1"] == 1.0
    assert m["mrr"] == 1.0


def test_per_persona_groups_by_persona():
    # 同一ペルソナの行はまとめて平均される。
    ranked = [[0, 1], [0, 1], [1, 0]]
    relevant = [{0}, {0}, {0}]
    personas = ["A", "A", "B"]
    per_p = per_persona_metrics(ranked, relevant, personas, ks=[1])
    assert set(per_p) == {"A", "B"}
    assert per_p["A"]["mrr"] == 1.0  # A の 2 行はどちらも 1 位
    assert per_p["B"]["mrr"] == 0.5  # B は 2 位


def test_macro_differs_from_micro_when_imbalanced():
    # A が 3 行（完璧）/ B が 1 行（4 位）。マイクロは A に偏るが、マクロは等重み。
    ranked = [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [1, 2, 3, 0]]
    relevant = [{0}, {0}, {0}, {0}]
    personas = ["A", "A", "A", "B"]

    micro = ir_metrics(ranked, relevant, ks=[1])  # (1+1+1+0.25)/4 = 0.8125
    assert math.isclose(micro["mrr"], 0.8125)

    summary = macro_summary(per_persona_metrics(ranked, relevant, personas, ks=[1]))
    # マクロは (A=1.0 + B=0.25)/2 = 0.625。頻出 A への偏りが消える。
    assert math.isclose(summary["macro"]["mrr"], 0.625)
    assert summary["n_personas"] == 2
    assert math.isclose(summary["per_persona"]["A"]["mrr"], 1.0)
    assert math.isclose(summary["per_persona"]["B"]["mrr"], 0.25)


def test_macro_summary_spread():
    ranked = [[0, 1], [1, 0]]
    relevant = [{0}, {0}]
    personas = ["A", "B"]  # A: mrr=1.0, B: mrr=0.5
    summary = macro_summary(per_persona_metrics(ranked, relevant, personas, ks=[1]))
    sp = summary["spread"]["mrr"]
    assert math.isclose(sp["min"], 0.5)
    assert math.isclose(sp["max"], 1.0)
    # mean=0.75, var=((0.25)^2+(0.25)^2)/2=0.0625, std=0.25
    assert math.isclose(sp["std"], 0.25)


def test_bootstrap_ci_deterministic_and_bounded():
    values = [0.25, 1.0]
    lo1, hi1 = _bootstrap_ci(values, resamples=500, seed=1)
    lo2, hi2 = _bootstrap_ci(values, resamples=500, seed=1)
    assert (lo1, hi1) == (lo2, hi2)  # 同 seed なら再現
    assert 0.25 <= lo1 <= hi1 <= 1.0  # CI は値域内
    # seed を変えると（一般に）変わりうる。少なくとも値域は保つ。
    lo3, hi3 = _bootstrap_ci(values, resamples=500, seed=999)
    assert 0.25 <= lo3 <= hi3 <= 1.0


def test_bootstrap_ci_single_value_degenerates():
    assert _bootstrap_ci([0.7], resamples=100, seed=0) == (0.7, 0.7)


def test_macro_summary_with_ci_keys():
    ranked = [[0, 1], [1, 0]]
    relevant = [{0}, {0}]
    personas = ["A", "B"]
    summary = macro_summary(
        per_persona_metrics(ranked, relevant, personas, ks=[1]), ci_resamples=200, seed=0
    )
    sp = summary["spread"]["mrr"]
    assert "ci_low" in sp and "ci_high" in sp
    assert sp["ci_low"] <= sp["ci_high"]
