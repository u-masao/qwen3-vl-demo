"""plots.py の純ロジック（描画を伴わない部分）の単体テスト（GPU・matplotlib 不要）。

描画（Figure 生成）は ``make smoke`` 側で担保し、ここではメトリクスキーの整形・並べ替え・
trainer_state のパースなど、決定的な純関数だけを検証する（既存 test_figures.py と同じ方針）。
"""

from __future__ import annotations

from qwen3vl_demo import plots


def test_strip_prefix():
    assert plots.strip_prefix(plots.METRIC_PREFIX + "ndcg@10") == "ndcg@10"
    assert plots.strip_prefix("ndcg@10") == "ndcg@10"  # 接頭辞が無ければそのまま


def test_pattern_label():
    assert plots.pattern_label("embed=ft+rerank=base") == "ft+base"


def test_metric_sort_key_orders_family_then_k():
    keys = ["mrr@10", "recall@5", "ndcg@10", "recall@1", "accuracy@1"]
    ordered = sorted(keys, key=plots.metric_sort_key)
    assert ordered == ["ndcg@10", "recall@1", "recall@5", "accuracy@1", "mrr@10"]


def test_ordered_patterns_known_first_unknown_last():
    metrics = {
        "embed=ft+rerank=ft": {},
        "embed=base+rerank=base": {},
        "embed=x+rerank=y": {},  # 未知パターン
    }
    ordered = plots.ordered_patterns(metrics)
    assert ordered[0] == "embed=base+rerank=base"  # PATTERN_ORDER 準拠で先頭
    assert ordered[-1] == "embed=x+rerank=y"  # 未知は末尾


def test_ordered_metric_keys_union_sorted():
    metrics = {"p1": {"ndcg@10": 0.1, "mrr": 0.2}, "p2": {"recall@5": 0.3}}
    keys = plots.ordered_metric_keys(metrics)
    assert set(keys) == {"ndcg@10", "mrr", "recall@5"}
    assert keys[0] == "ndcg@10"  # ndcg は priority 0


def test_bar_colors_by_sign():
    assert plots.bar_colors([0.5, -0.5, 0.0]) == [
        plots.COLOR_FT,
        plots.COLOR_BASE,
        plots.COLOR_ZERO,
    ]


def test_parse_trainer_state_splits_loss_and_eval():
    state = {
        "log_history": [
            {"step": 10, "loss": 0.9},
            {"step": 20, "loss": 0.7},
            {"step": 20, "eval_ndcg@10": 0.5, "eval_loss": 0.6},
        ],
        "best_global_step": 20,
    }
    curve = plots.parse_trainer_state(state)
    assert curve["loss"] == [(10, 0.9), (20, 0.7)]
    assert curve["eval"]["eval_ndcg@10"] == [(20, 0.5)]
    assert curve["best_step"] == 20


def test_parse_trainer_state_empty():
    curve = plots.parse_trainer_state({})
    assert curve["loss"] == []
    assert curve["eval"] == {}


def test_parse_trainer_state_sorts_by_step():
    state = {"log_history": [{"step": 30, "loss": 0.5}, {"step": 10, "loss": 0.9}]}
    curve = plots.parse_trainer_state(state)
    assert curve["loss"] == [(10, 0.9), (30, 0.5)]
