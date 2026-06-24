"""検索メトリクスの純関数（GPU 不要・単体テスト対象）。

このプロジェクトの eval は「クエリ＝ペルソナ名」で、ペルソナは数種類しかない。行（画像）単位で
平均（マイクロ平均）すると、画像枚数の多い頻出ペルソナが指標を支配して**偏る**（信頼できない）。
そこで **per-persona マクロ平均**（各ペルソナを等重みで平均）と、ペルソナ間の**ばらつき**
（std / min / max・ブートストラップ信頼区間）を出すためのユーティリティをここに集約する。

`evaluate.py` / `rerank.py` の双方から import して、集計ロジックの重複を避ける。
"""

from __future__ import annotations

import math
import random


def ir_metrics(
    ranked_lists: list[list[int]],
    relevant_sets: list[set[int]],
    ks: list[int],
) -> dict[str, float]:
    """ランキング結果から Recall@k / NDCG@k / MRR を計算する（クエリ平均・純粋関数）。

    Args:
        ranked_lists: 各クエリの、関連度降順に並んだ文書インデックス列。
        relevant_sets: 各クエリの正解文書インデックス集合。
        ks: Recall / NDCG を測る上位件数のリスト。

    Returns:
        ``{"recall@k": ..., "ndcg@k": ..., "mrr": ...}`` の dict（与えたクエリ集合での平均）。
    """
    n = max(1, len(ranked_lists))
    out: dict[str, float] = {}

    for k in ks:
        recall_sum = 0.0
        ndcg_sum = 0.0
        for ranked, rel in zip(ranked_lists, relevant_sets, strict=False):
            if not rel:
                continue
            topk = ranked[:k]
            hits = sum(1 for d in topk if d in rel)
            recall_sum += hits / len(rel)
            # 2 値関連度の DCG / IDCG。
            dcg = sum(1.0 / math.log2(pos + 1) for pos, d in enumerate(topk, start=1) if d in rel)
            ideal = sum(1.0 / math.log2(p + 1) for p in range(1, min(k, len(rel)) + 1))
            ndcg_sum += (dcg / ideal) if ideal > 0 else 0.0
        out[f"recall@{k}"] = recall_sum / n
        out[f"ndcg@{k}"] = ndcg_sum / n

    # MRR: 最初に現れた正解の逆順位（ランキング全体を対象）。
    mrr_sum = 0.0
    for ranked, rel in zip(ranked_lists, relevant_sets, strict=False):
        for pos, d in enumerate(ranked, start=1):
            if d in rel:
                mrr_sum += 1.0 / pos
                break
    out["mrr"] = mrr_sum / n
    return out


def per_persona_metrics(
    ranked_lists: list[list[int]],
    relevant_sets: list[set[int]],
    personas: list[str],
    ks: list[int],
) -> dict[str, dict[str, float]]:
    """行（クエリ）をペルソナでグループ化し、ペルソナごとに :func:`ir_metrics` を計算する。

    ``personas[i]`` が行 ``i`` のペルソナ名。同一ペルソナの行をまとめて平均するので、戻り値は
    ``{persona: {"recall@k":..., "ndcg@k":..., "mrr":...}}``。
    """
    groups: dict[str, list[int]] = {}
    for i, p in enumerate(personas):
        groups.setdefault(p, []).append(i)

    out: dict[str, dict[str, float]] = {}
    for persona, idxs in groups.items():
        rl = [ranked_lists[i] for i in idxs]
        rs = [relevant_sets[i] for i in idxs]
        out[persona] = ir_metrics(rl, rs, ks)
    return out


def _bootstrap_ci(
    values: list[float], resamples: int, seed: int, alpha: float = 0.05
) -> tuple[float, float]:
    """ペルソナ単位スカラ列の平均に対する、ブートストラップ信頼区間 (lo, hi) を返す。

    ``resamples`` 回、値を復元抽出して平均を取り、両側 ``alpha`` のパーセンタイルを返す。
    ``seed`` で決定的（同 seed なら再現）。値が空/1 件のときは縮退する。
    """
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"))
    if n == 1 or resamples <= 0:
        return (values[0], values[0])
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_idx = max(0, int((alpha / 2) * resamples))
    hi_idx = min(resamples - 1, int((1 - alpha / 2) * resamples))
    return (means[lo_idx], means[hi_idx])


def macro_summary(
    per_persona: dict[str, dict[str, float]],
    ci_resamples: int = 0,
    seed: int = 0,
) -> dict:
    """per-persona メトリクスから、マクロ平均と ペルソナ間ばらつきをまとめる。

    Returns:
        ``{"macro": {metric: mean}, "spread": {metric: {std,min,max[,ci_low,ci_high]}},
        "per_persona": {...}, "n_personas": int}``。``macro`` は :func:`ir_metrics` と同じ
        キー（recall@k / ndcg@k / mrr）なので、フラットなメトリクス dict として後方互換に使える。
    """
    personas = sorted(per_persona)
    metric_keys = sorted({k for m in per_persona.values() for k in m})

    macro: dict[str, float] = {}
    spread: dict[str, dict[str, float]] = {}
    for key in metric_keys:
        vals = [per_persona[p][key] for p in personas]
        mean = sum(vals) / len(vals) if vals else 0.0
        macro[key] = mean
        var = sum((v - mean) ** 2 for v in vals) / len(vals) if vals else 0.0
        s: dict[str, float] = {"std": var**0.5, "min": min(vals), "max": max(vals)}
        if ci_resamples > 0:
            lo, hi = _bootstrap_ci(vals, ci_resamples, seed)
            s["ci_low"], s["ci_high"] = lo, hi
        spread[key] = s

    return {
        "macro": macro,
        "spread": spread,
        "per_persona": per_persona,
        "n_personas": len(personas),
    }
