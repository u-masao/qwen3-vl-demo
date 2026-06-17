"""distill.py の純粋関数（GPU/モデル不要）の単体テスト。

ハードネガティブ列のクエリ単位への畳み込み（group_negatives）、リランカー teacher の
マージン行（build_margin_rows）、oracle teacher の soft relevance 行（build_oracle_rows）を
検証する。属性復元（preference.fragments_to_attributes）の往復も併せて確認する。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from qwen3vl_demo.config import Config
from qwen3vl_demo.distill import (
    _resolve_student_id,
    build_margin_rows,
    build_oracle_rows,
    group_negatives,
)
from qwen3vl_demo.preference import (
    attributes_to_fragments,
    build_model,
    fragments_to_attributes,
    relevance_score,
)


def _pairs():
    # 2 クエリ。各 1 正例 + 2 負例（mine_hard_negatives の出力形式）。
    return [
        (0, 0, 1.0),
        (0, 3, 0.0),
        (0, 4, 0.0),
        (1, 1, 1.0),
        (1, 5, 0.0),
        (1, 2, 0.0),
    ]


# --- group_negatives --------------------------------------------------------
def test_group_negatives_basic():
    grouped = group_negatives(_pairs())
    assert grouped == [(0, 0, [3, 4]), (1, 1, [5, 2])]


def test_group_negatives_sorted_by_query():
    # 入力順がばらけても query 昇順で返る。
    pairs = [(1, 1, 1.0), (1, 0, 0.0), (0, 0, 1.0), (0, 2, 0.0)]
    grouped = group_negatives(pairs)
    assert [q for q, _, _ in grouped] == [0, 1]


def test_group_negatives_query_without_negatives():
    grouped = group_negatives([(0, 0, 1.0)])
    assert grouped == [(0, 0, [])]


# --- build_margin_rows ------------------------------------------------------
def test_build_margin_rows_computes_teacher_margin():
    grouped = [(0, 0, [3, 4])]
    scores = {(0, 0): 2.0, (0, 3): 0.5, (0, 4): -1.0}
    rows = build_margin_rows(grouped, scores)
    # (query, pos, neg, margin=s_pos - s_neg)
    assert rows == [(0, 0, 3, 1.5), (0, 0, 4, 3.0)]


def test_build_margin_rows_skips_queries_without_negatives():
    rows = build_margin_rows([(0, 0, [])], {(0, 0): 1.0})
    assert rows == []


# --- build_oracle_rows ------------------------------------------------------
def test_build_oracle_rows_matches_relevance_score():
    model = build_model(seed=0)
    personas = model.personas()
    # doc ごとに属性 → プロンプト文を組み立てる（build_captions_preference と同形式）。
    attrs_by_doc = {
        0: [1, 1, 0, 0, 0, 0, 1],
        1: [0, 0, 1, 1, 0, 1, 0],
        2: [1, 0, 1, 0, 1, 0, 1],
    }
    texts = [
        "a photo of a cat, " + ", ".join(attributes_to_fragments(model, attrs_by_doc[i]))
        for i in range(3)
    ]
    # クエリ 0 のペルソナは personas[0]、doc は pos=0 と neg=[1, 2]。
    grouped = [(0, 0, [1, 2])]
    rows = build_oracle_rows(grouped, [personas[0]], texts, model)

    assert [doc for _, doc, _ in rows] == [0, 1, 2]
    for _, doc, label in rows:
        expected = relevance_score(model, personas[0], attrs_by_doc[doc])
        assert label == pytest.approx(expected)
        assert 0.0 <= label <= 1.0


# --- _resolve_student_id（蒸留先 student の初期化元解決）---------------------
def test_resolve_student_id_none_is_self_distill():
    # 既定（None）＝ベース埋め込みからの自己蒸留。
    cfg = Config()
    cfg.distill.student_model = None
    assert _resolve_student_id(cfg) == cfg.embedding.model_id


def test_resolve_student_id_empty_string_is_self_distill():
    # 空文字（DVC が "none" を None に潰す前後の取りこぼし対策）もベース扱い。
    cfg = Config()
    cfg.distill.student_model = ""
    assert _resolve_student_id(cfg) == cfg.embedding.model_id


def test_resolve_student_id_ft_uses_model_path():
    # "ft" は FT 済み埋め込み成果物（絶対パスに解決した model_path）から継続蒸留する。
    cfg = Config()
    cfg.distill.student_model = "ft"
    resolved = _resolve_student_id(cfg)
    assert resolved == str(cfg.model_path)
    assert Path(resolved).is_absolute()


def test_resolve_student_id_arbitrary_id_passthrough():
    # 任意の HF ID／ローカルパスはそのまま使う（小型 cross-modal 埋め込みへ圧縮）。
    cfg = Config()
    cfg.distill.student_model = "sentence-transformers/clip-ViT-B-32"
    assert _resolve_student_id(cfg) == "sentence-transformers/clip-ViT-B-32"


# --- fragments_to_attributes（往復）-----------------------------------------
def test_fragments_to_attributes_roundtrip():
    model = build_model(seed=0)
    for attrs in ([0] * len(model.axes), [1] * len(model.axes), [1, 0, 1, 0, 1, 0, 1]):
        text = "a photo of a cat, " + ", ".join(attributes_to_fragments(model, attrs))
        assert fragments_to_attributes(model, text) == attrs


def test_fragments_to_attributes_raises_on_missing_axis():
    model = build_model(seed=0)
    with pytest.raises(ValueError):
        fragments_to_attributes(model, "a photo of a cat with no fragments")
