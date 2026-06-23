"""figures.py のサンプリング補助関数の単体テスト（純 Python・モデル不要）。

画像描画（matplotlib / モデルロード）には触れず、図に「どの行を選ぶか」を決める
純粋ロジックだけを検証する。``ds`` は ``enumerate`` できて各要素が dict ならよいので、
データセットの代わりに dict のリストを渡す。
"""

from __future__ import annotations

from qwen3vl_demo.figures import (
    _appeal_components,
    _confusion_matrix,
    _count_personas,
    _interaction_edges,
    _pick_query_indices,
    _select_grid_indices,
    _select_latest_checkpoint,
)


def _fake_ds():
    # 3 カテゴリ・2 ペルソナの小さな擬似データセット。
    rows = []
    for cat in ("animal", "vehicle", "food"):
        for j in range(3):
            persona = "user_alpha" if j % 2 == 0 else "user_beta"
            rows.append({"category": cat, "persona": persona, "anchor": f"{cat}-{j}"})
    return rows


def test_grid_respects_count():
    ds = _fake_ds()
    assert len(_select_grid_indices(ds, 5)) == 5


def test_grid_caps_at_dataset_size():
    ds = _fake_ds()
    picked = _select_grid_indices(ds, 100)
    assert len(picked) == len(ds)
    assert len(set(picked)) == len(picked)  # 重複なし


def test_grid_spreads_across_categories():
    ds = _fake_ds()
    picked = _select_grid_indices(ds, 3)
    cats = {ds[i]["category"] for i in picked}
    # 先頭 3 件はラウンドロビンで 3 カテゴリすべてから 1 件ずつ拾うはず。
    assert cats == {"animal", "vehicle", "food"}


def test_pick_query_indices_distinct_personas():
    ds = _fake_ds()
    picked = _pick_query_indices(ds, 5)
    personas = [ds[i]["persona"] for i in picked]
    assert personas == list(dict.fromkeys(personas))  # 重複なし
    assert len(picked) <= 2  # ペルソナは 2 種類しかない


def test_pick_query_indices_limit():
    ds = _fake_ds()
    assert len(_pick_query_indices(ds, 1)) == 1


# --- 学習曲線・嗜好図・混同行列の補助ロジック ------------------------------


def test_select_latest_checkpoint_picks_max():
    names = ["checkpoint-10", "checkpoint-250", "checkpoint-50"]
    assert _select_latest_checkpoint(names) == "checkpoint-250"


def test_select_latest_checkpoint_ignores_non_numeric():
    names = ["checkpoint-best", "checkpoint-30", "foo"]
    assert _select_latest_checkpoint(names) == "checkpoint-30"


def test_select_latest_checkpoint_empty():
    assert _select_latest_checkpoint([]) is None


def test_count_personas_orders_by_count_without_order():
    counts = _count_personas(["a", "b", "a", "a", "b", "c"])
    assert counts == {"a": 3, "b": 2, "c": 1}


def test_count_personas_with_order_fills_absent_with_zero():
    counts = _count_personas(["a", "a", "c"], order=["a", "b", "c"])
    assert counts == {"a": 2, "b": 0, "c": 1}
    assert list(counts) == ["a", "b", "c"]  # 与えた順序を保つ


def test_count_personas_appends_unknown_persona():
    counts = _count_personas(["a", "z"], order=["a"])
    assert counts["z"] == 1  # order に無いペルソナも取りこぼさない


def test_interaction_edges_from_dict():
    model = {"interactions": {"p": [[0.0, 2.0, -2.0], [1.0, 4.0, 1.5]]}}
    assert _interaction_edges(model, "p") == [(0, 2, -2.0), (1, 4, 1.5)]
    assert _interaction_edges(model, "missing") == []


def test_confusion_matrix_counts_retrieved_personas():
    order = ["a", "b"]
    doc_personas = ["a", "a", "b"]  # doc 0,1 が a / doc 2 が b
    ranked = [[0, 2], [1]]  # クエリ a は doc0,2 / クエリ b は doc1 を取得
    queries = ["a", "b"]
    # a 行: a を 1（doc0）・b を 1（doc2）/ b 行: a を 1（doc1）
    assert _confusion_matrix(ranked, queries, doc_personas, order) == [[1, 1], [1, 0]]


def test_appeal_components_sum_matches_appeal():
    # 寄与分解（ノイズ除く）の和が、preference.appeal からノイズを引いた値に一致する。
    from qwen3vl_demo.preference import _noise, appeal, build_model

    model = build_model(seed=1)
    persona = model.personas()[0]
    attrs = [1, 0, 1, 0, 1, 0, 1]
    lin, inter, pop = _appeal_components(model, persona, attrs)
    expected = appeal(model, persona, attrs) - _noise(model, persona, attrs)
    assert abs((lin + inter + pop) - expected) < 1e-9
