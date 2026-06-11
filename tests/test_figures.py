"""figures.py のサンプリング補助関数の単体テスト（純 Python・モデル不要）。

画像描画（matplotlib / モデルロード）には触れず、図に「どの行を選ぶか」を決める
純粋ロジックだけを検証する。``ds`` は ``enumerate`` できて各要素が dict ならよいので、
データセットの代わりに dict のリストを渡す。
"""

from __future__ import annotations

from qwen3vl_demo.figures import _pick_query_indices, _select_grid_indices


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
