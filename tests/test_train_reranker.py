"""train_reranker.build_pair_indices（ネガティブマイニング）の単体テスト。"""

from __future__ import annotations

from qwen3vl_demo.train_reranker import build_pair_indices


def _categories():
    # 2 カテゴリ × 3 件ずつの単純なレイアウト。
    return ["animal", "animal", "animal", "food", "food", "food"]


def test_one_positive_per_query():
    cats = _categories()
    pairs = build_pair_indices(cats, num_negatives=2, seed=0)
    positives = [(q, d) for q, d, label in pairs if label == 1.0]
    # 各クエリ i にちょうど 1 件の正例 (i, i)。
    assert sorted(positives) == [(i, i) for i in range(len(cats))]


def test_negative_count_per_query():
    cats = _categories()
    pairs = build_pair_indices(cats, num_negatives=2, seed=0)
    for i in range(len(cats)):
        negs = [(q, d) for q, d, label in pairs if label == 0.0 and q == i]
        assert len(negs) == 2
        assert all(d != i for _, d in negs)  # 自分自身は負例にしない


def test_negatives_prefer_different_category():
    cats = _categories()
    # num_negatives=3 でも、別カテゴリ（3 件）で賄えるので全負例が別カテゴリのはず。
    pairs = build_pair_indices(cats, num_negatives=3, seed=1)
    for q, d, label in pairs:
        if label == 0.0:
            assert cats[d] != cats[q]


def test_deterministic():
    cats = _categories()
    a = build_pair_indices(cats, num_negatives=2, seed=5)
    b = build_pair_indices(cats, num_negatives=2, seed=5)
    assert a == b


def test_single_row_has_no_negatives():
    pairs = build_pair_indices(["animal"], num_negatives=3, seed=0)
    assert pairs == [(0, 0, 1.0)]
