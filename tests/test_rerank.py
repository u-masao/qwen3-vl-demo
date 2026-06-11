"""rerank の純 Python 部（メトリクス・正解集合・順位）の単体テスト。"""

from __future__ import annotations

import math

from qwen3vl_demo.rerank import _build_relevant, _metrics_for, _rank_of


def test_rank_of():
    assert _rank_of(3, [5, 3, 1]) == 2
    assert _rank_of(9, [5, 3, 1]) is None


def test_build_relevant_strict():
    # 正解は「同一ペルソナの全文書」（マルチポジティブ）。
    ds = [
        {"persona": "p1", "category": "a"},
        {"persona": "p2", "category": "b"},
        {"persona": "p1", "category": "a"},
    ]
    rel = _build_relevant(ds, relevant_same_category=False)
    # p1 = {0,2}、p2 = {1}。
    assert rel == [{0, 2}, {1}, {0, 2}]


def test_build_relevant_same_category():
    # relevant_same_category=True ではペルソナ集合に同一カテゴリ集合を和で追加する。
    ds = [
        {"persona": "p1", "category": "a"},
        {"persona": "p2", "category": "a"},
        {"persona": "p2", "category": "b"},
    ]
    rel = _build_relevant(ds, relevant_same_category=True)
    # ペルソナ: p1={0}, p2={1,2} / カテゴリ: a={0,1}, b={2}。
    # row0: {0} | {0,1} = {0,1} / row1: {1,2} | {0,1} = {0,1,2} / row2: {1,2} | {2} = {1,2}
    assert rel == [{0, 1}, {0, 1, 2}, {1, 2}]


def test_metrics_perfect_ranking():
    # 各クエリの正解が 1 位 -> recall/ndcg=1, mrr=1。
    ranked = [[0, 1, 2], [1, 0, 2]]
    relevant = [{0}, {1}]
    m = _metrics_for(ranked, relevant, ks=[1, 3])
    assert m["recall@1"] == 1.0
    assert m["ndcg@1"] == 1.0
    assert m["mrr"] == 1.0


def test_metrics_target_at_second_place():
    # 正解が 2 位 -> recall@1=0, mrr=0.5, ndcg@2 = (1/log2(3)) / (1/log2(2))。
    ranked = [[9, 0, 1]]
    relevant = [{0}]
    m = _metrics_for(ranked, relevant, ks=[1, 2])
    assert m["recall@1"] == 0.0
    assert m["recall@2"] == 1.0
    assert m["mrr"] == 0.5
    expected_ndcg2 = (1.0 / math.log2(3)) / (1.0 / math.log2(2))
    assert math.isclose(m["ndcg@2"], expected_ndcg2, rel_tol=1e-9)


def test_metrics_multi_relevant_recall():
    # 正解が 2 件、top-2 に 1 件だけ含まれる -> recall@2 = 0.5。
    ranked = [[0, 9, 1]]
    relevant = [{0, 1}]
    m = _metrics_for(ranked, relevant, ks=[2])
    assert m["recall@2"] == 0.5
