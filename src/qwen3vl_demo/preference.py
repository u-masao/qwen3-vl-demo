"""人間の嗜好を模した「裏設定（潜在嗜好）」モデル。

このモジュールは、ペルソナ検索タスクの **正解（relevance）の作り方** を一手に引き受ける
単一の真実（SSOT）。``generate_data`` / ``evaluate`` / ``rerank`` / ``train_reranker`` は
すべてこの嗜好モデルを参照し、`data/preference_model.json` に保存された同一の ground-truth
から graded relevance を計算する。

設計の狙い
----------
従来のペルソナタスク（subject の恣意的割当・二値 relevance）は、(1) embed FT が暗記で
飽和し、(2) リランカー FT の伸びしろが消える、という問題があった。本モデルは **人間の嗜好の
構造** を写し取ることで、これを構造的に解消する。

1. **低次元の潜在因子** … 嗜好は少数の潜在軸（warmth / era / ornament / mood）に載る。
2. **アーキタイプの混合** … 各ペルソナは少数の共有アーキタイプ（型）の混合。共有構造が
   あるため、将来の未知ペルソナ few-shot 汎化（v2）の足場になる。
3. **確率的・段階的** … relevance は二値でなく連続値（graded）。境界がゆらぐ＝「緩い一貫性」。
4. **非加法的な交互作用（最重要）** … 「warm も ornate も好き、でも warm かつ ornate は嫌い」
   のような AND/NOT の相互作用。これは内積で候補をスコアする bi-encoder（埋め込み）が表現
   しづらく、クエリと候補を結合して見る cross-encoder（リランカー）が得意とする。つまり
   **リランカーの伸びしろは、この交互作用に由来する**。`gamma`（交互作用強度）で大きさを制御し、
   ``gamma=0`` なら加法的＝リランカー伸びしろ≈0（旧タスクの再現）、``gamma>0`` で伸びしろが出る。
5. **人気バイアス** … 平均的な嗜好に沿う候補は万人受けする（`lam` で制御）。

クエリは opaque トークン（``"user_alpha"`` 等）のままなので、ベースモデルは
トークンと嗜好の対応を知らず ``base ≈ random`` が保たれ、FT の効果を測れる。

属性の表現
----------
1 候補（画像）の潜在属性 ``a`` は軸ごとの二値ベクトル（0/1）。線形項では中心化した
``2a-1 ∈ {-1,+1}`` を使い「好む属性の有無」と「嫌う属性の有無」を対称に評価する。
交互作用項は生の ``a`` の AND（``a_i*a_j``）に作用する。
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

# --- 潜在嗜好軸 -------------------------------------------------------------
# 軸の並び順がベクトルのインデックスを定義する。各軸は二値（0/1）。
AXES: list[str] = ["warmth", "era", "ornament", "mood"]

# 各軸の値 → 画像生成プロンプトに差し込む語片。(value=0 の語片, value=1 の語片)。
FRAGMENTS: dict[str, tuple[str, str]] = {
    "warmth": ("cool-toned", "warm-toned"),
    "era": ("modern", "vintage"),
    "ornament": ("minimalist, clean", "ornate, intricately detailed"),
    "mood": ("bright, airy lighting", "moody, dim lighting"),
}

# --- アーキタイプ（共有される「型」）----------------------------------------
# 各アーキタイプは軸上の嗜好ベクトル（+好む / -嫌う / 0中立）。ペルソナはこれらの混合。
ARCHETYPES: dict[str, list[float]] = {
    #                 warmth  era   ornament  mood
    "retro_warm": [1.0, 1.0, 0.0, 0.0],  # 暖色・ヴィンテージ好き
    "modern_minimal": [-1.0, -1.0, -1.0, 0.0],  # 寒色・モダン・ミニマル好き
    "ornate_moody": [0.0, 0.0, 1.0, 1.0],  # 装飾的・moody 好き
    "bright_airy": [0.5, 0.0, -1.0, -1.0],  # ミニマル・明るい好き（やや暖色）
}

# --- ペルソナ＝アーキタイプの混合（convex weights, 合計 1）------------------
# わざと重なりを持たせる（共有構造＝「似て非なる」候補を生み、リランカーの仕事を作る）。
PERSONA_MIX: dict[str, dict[str, float]] = {
    "user_alpha": {"retro_warm": 0.7, "ornate_moody": 0.3},
    "user_beta": {"modern_minimal": 0.7, "bright_airy": 0.3},
    "user_gamma": {"ornate_moody": 0.6, "retro_warm": 0.4},
    "user_delta": {"bright_airy": 0.6, "modern_minimal": 0.4},
    "user_epsilon": {"retro_warm": 0.5, "modern_minimal": 0.5},  # 相反する型の混合
    "user_zeta": {"ornate_moody": 0.5, "bright_airy": 0.5},
    "user_eta": {"retro_warm": 0.34, "modern_minimal": 0.33, "ornate_moody": 0.33},
}

# --- 非加法的な交互作用 ------------------------------------------------------
# persona -> [(axis_i, axis_j, coef), ...]。appeal に coef * (a_i AND a_j) を gamma 倍で加える。
# 負の coef は「i も j も単体では好きだが、両方そろうと嫌い」という非単調な嗜好を作る
# （線形＝加法モデルでは表現できない＝リランカーの領分）。
INTERACTIONS: dict[str, list[tuple[int, int, float]]] = {
    "user_alpha": [(0, 2, -1.0)],  # warm かつ ornate は過剰で嫌い
    "user_beta": [(1, 3, +1.0)],  # vintage かつ moody の意外な組合せが好き
    "user_gamma": [(0, 3, -1.0)],  # warm かつ moody は嫌い
    "user_delta": [(2, 1, +1.0)],  # ornate かつ vintage が好き
    "user_epsilon": [(0, 1, +1.0)],  # warm かつ vintage で相反を解消
    "user_zeta": [(2, 0, -1.0)],  # ornate かつ warm は嫌い
    "user_eta": [(1, 2, +1.0), (0, 3, -1.0)],
}


@dataclass
class PreferenceModel:
    """解決済みの嗖好モデル（JSON 直列化可能な数値表現）。

    ``build_model`` でコード中のテンプレ（アーキタイプ・混合・交互作用）＋スカラ knob から
    構築し、``data/preference_model.json`` に保存して全ステージで共有する。
    """

    axes: list[str]
    fragments: dict[str, list[str]]  # axis -> [frag0, frag1]
    archetypes: dict[str, list[float]]
    persona_mix: dict[str, dict[str, float]]
    persona_pref: dict[str, list[float]]  # persona -> θ_p（軸上の嗜好ベクトル）
    global_pref: list[float]  # アーキタイプ平均＝人気バイアスの基準
    interactions: dict[str, list[list[float]]]  # persona -> [[i, j, coef], ...]
    gamma: float  # 交互作用強度（0=加法のみ）
    lam: float  # 人気バイアス強度
    sigma: float  # 個人ノイズ振幅（決定的）
    temperature: float  # appeal -> 確率へのシグモイド温度
    sharpness: float  # 属性サンプリングの鋭さ（高いほど嗜好に忠実＝一貫性が強い）
    threshold: float  # relevance を二値化する閾値（recall/MRR 用）
    seed: int  # ノイズ用シード

    def personas(self) -> list[str]:
        """ペルソナ名の一覧（決定的な順序）。"""
        return list(self.persona_pref.keys())


# --- 構築 -------------------------------------------------------------------
def _resolve_persona_pref(
    archetypes: dict[str, list[float]], mix: dict[str, dict[str, float]], n_axes: int
) -> dict[str, list[float]]:
    """各ペルソナの嗖好ベクトル θ_p = Σ_k w_{p,k} · archetype_k を計算する。"""
    out: dict[str, list[float]] = {}
    for persona, weights in mix.items():
        vec = [0.0] * n_axes
        for arch_name, w in weights.items():
            arch = archetypes[arch_name]
            for i in range(n_axes):
                vec[i] += w * arch[i]
        out[persona] = vec
    return out


def _mean_vectors(vectors: list[list[float]], n_axes: int) -> list[float]:
    if not vectors:
        return [0.0] * n_axes
    out = [0.0] * n_axes
    for v in vectors:
        for i in range(n_axes):
            out[i] += v[i]
    return [x / len(vectors) for x in out]


def build_model(
    *,
    gamma: float = 1.0,
    lam: float = 0.3,
    sigma: float = 0.1,
    temperature: float = 1.0,
    sharpness: float = 2.0,
    threshold: float = 0.5,
    seed: int = 42,
) -> PreferenceModel:
    """コードのテンプレ＋スカラ knob から :class:`PreferenceModel` を構築する。

    knob（特に ``gamma``）が難易度とリランカー伸びしろを制御する中心。
    """
    n = len(AXES)
    persona_pref = _resolve_persona_pref(ARCHETYPES, PERSONA_MIX, n)
    global_pref = _mean_vectors(list(ARCHETYPES.values()), n)
    interactions = {
        p: [[float(i), float(j), float(c)] for (i, j, c) in lst] for p, lst in INTERACTIONS.items()
    }
    return PreferenceModel(
        axes=list(AXES),
        fragments={ax: list(FRAGMENTS[ax]) for ax in AXES},
        archetypes={k: list(v) for k, v in ARCHETYPES.items()},
        persona_mix={p: dict(w) for p, w in PERSONA_MIX.items()},
        persona_pref=persona_pref,
        global_pref=global_pref,
        interactions=interactions,
        gamma=gamma,
        lam=lam,
        sigma=sigma,
        temperature=temperature,
        sharpness=sharpness,
        threshold=threshold,
        seed=seed,
    )


# --- 数値ヘルパ -------------------------------------------------------------
def _sigmoid(x: float) -> float:
    # オーバーフローを避けた安定なシグモイド。
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def _centered(attrs: list[int]) -> list[float]:
    """二値属性 {0,1} を {-1,+1} に中心化（線形項・人気項で対称評価するため）。"""
    return [2.0 * v - 1.0 for v in attrs]


def _noise(model: PreferenceModel, persona: str, attrs: list[int]) -> float:
    """(persona, attrs) に対して決定的な個人ノイズ。同一属性なら同一値（再現性）。"""
    if model.sigma == 0.0:
        return 0.0
    key = f"{model.seed}:{persona}:{','.join(str(v) for v in attrs)}"
    digest = hashlib.sha256(key.encode()).digest()
    u = int.from_bytes(digest[:8], "big") / float(1 << 64)  # [0, 1)
    return model.sigma * (u - 0.5)


# --- 嗖好スコア -------------------------------------------------------------
def appeal(model: PreferenceModel, persona: str, attrs: list[int]) -> float:
    """ペルソナ ``persona`` が候補（属性 ``attrs``）をどれだけ好むかの生スコア。

    appeal = θ_p·(2a-1)  +  γ·Σ coef·(a_i AND a_j)  +  λ·θ_global·(2a-1)  +  ε
    """
    centered = _centered(attrs)
    score = _dot(model.persona_pref[persona], centered)  # 線形：潜在嗖好
    for tri in model.interactions.get(persona, []):  # 非加法：AND 交互作用
        i, j, c = int(tri[0]), int(tri[1]), tri[2]
        score += model.gamma * c * (attrs[i] * attrs[j])
    score += model.lam * _dot(model.global_pref, centered)  # 人気バイアス
    score += _noise(model, persona, attrs)  # 個人ノイズ
    return score


def relevance_score(model: PreferenceModel, persona: str, attrs: list[int]) -> float:
    """appeal をシグモイドで [0,1] の graded relevance に変換する。"""
    return _sigmoid(appeal(model, persona, attrs) / model.temperature)


def is_relevant(model: PreferenceModel, persona: str, attrs: list[int]) -> bool:
    """二値 relevance（recall/MRR 用）。``relevance_score >= threshold`` なら正解。"""
    return relevance_score(model, persona, attrs) >= model.threshold


def graded_relevance(
    model: PreferenceModel, persona: str, corpus_attrs: list[list[int]]
) -> dict[int, float]:
    """コーパス全件に対する graded relevance（NDCG の gain）を返す（idx -> [0,1]）。"""
    return {idx: relevance_score(model, persona, attrs) for idx, attrs in enumerate(corpus_attrs)}


# --- 生成（属性サンプリング・プロンプト語片）-------------------------------
def sample_item_attributes(model: PreferenceModel, persona: str, rng: random.Random) -> list[int]:
    """ペルソナの嗖好分布から 1 候補の二値属性をサンプルする（緩い一貫性）。

    各軸で P(a_i=1) = sigmoid(sharpness · θ_{p,i})。sharpness が高いほど嗜好に忠実
    （= 一貫性が強い）。低いほど散漫になる。
    """
    theta = model.persona_pref[persona]
    return [1 if rng.random() < _sigmoid(model.sharpness * w) else 0 for w in theta]


def attributes_to_fragments(model: PreferenceModel, attrs: list[int]) -> list[str]:
    """二値属性を画像生成プロンプト用の語片リストに変換する。"""
    return [model.fragments[ax][attrs[i]] for i, ax in enumerate(model.axes)]


# --- 属性の (de)シリアライズ（HF dataset の 1 カラムに収める）---------------
def encode_attributes(attrs: list[int]) -> str:
    """属性ベクトルを dataset 格納用の文字列にする（"1,0,1,0"）。"""
    return ",".join(str(int(v)) for v in attrs)


def decode_attributes(s: str) -> list[int]:
    """``encode_attributes`` の逆。空文字は空リスト。"""
    s = s.strip()
    return [int(x) for x in s.split(",")] if s else []


# --- モデルの保存・読み込み（全ステージで同一 ground-truth を共有）---------
def save_model(model: PreferenceModel, path: str | Path) -> None:
    """嗖好モデルを JSON に保存する（``data/preference_model.json``）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(asdict(model), fh, ensure_ascii=False, indent=2)


def load_model(path: str | Path) -> PreferenceModel:
    """``save_model`` で書いた JSON から :class:`PreferenceModel` を復元する。"""
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    known = {f for f in PreferenceModel.__dataclass_fields__}  # type: ignore[attr-defined]
    return PreferenceModel(**{k: v for k, v in raw.items() if k in known})
