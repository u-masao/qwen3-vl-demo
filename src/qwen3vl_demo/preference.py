"""人間の嗜好を模した「裏設定（潜在嗜好）」モデル。

このモジュールは、ペルソナ検索タスク（``data.task = preference``）の **正解（relevance）の
作り方** を一手に引き受ける単一の真実（SSOT）。``generate_data`` が各画像の属性をサンプルし、
**その画像を最も好むペルソナ（argmax appeal）** をラベル（``persona`` 列）に付ける。これにより
評価・学習・リランクの下流コードは「persona 列の一致＝関連」という従来の仕組みのまま、一切変えずに
新しいタスクへ切り替えられる（既存の subject タスクと**並列**のバリアント）。

設計の狙い
----------
従来のペルソナタスク（subject の恣意的割当・二値 relevance）は、(1) embed FT が暗記で
飽和し、(2) リランカー FT の伸びしろが消える、という問題があった。本モデルは **人間の嗜好の
構造** を写し取ることで、これを構造的に解消する。

1. **低次元の潜在因子** … 嗜好は少数の潜在軸（warmth / era / ornament / mood / saturation /
   material / setting）に載る。
2. **アーキタイプの混合** … 各ペルソナは少数の共有アーキタイプ（型）の sparse な混合。共有構造が
   あるため、将来の未知ペルソナ few-shot 汎化（v2）の足場になる。
3. **確率的・段階的** … 属性は嗖好分布から確率的にサンプルされ、appeal は連続値。境界がゆらぐ
   ＝「緩い一貫性」。
4. **非加法的な交互作用（最重要）** … 「warm も ornate も単体では好き、でも warm かつ ornate は嫌い」
   のような AND/NOT の相互作用。これは内積で候補をスコアする bi-encoder（埋め込み）が表現しづらく、
   クエリと候補を結合して見る cross-encoder（リランカー）が得意とする。つまり **リランカーの伸びしろは、
   この交互作用に由来する**。``gamma``（交互作用強度）で大きさを制御し、``gamma=0`` なら加法的＝
   リランカー伸びしろ≈0（旧タスクの再現）、``gamma>0`` で伸びしろが出る、という難易度ノブになる。
5. **人気バイアス** … 平均的な嗖好に沿う候補は万人受けする（``lam`` で制御）。

クエリは opaque トークン（``"user_alpha"`` 等）のままなので、ベースモデルはトークンと嗖好の対応を
知らず ``base ≈ random`` が保たれ、FT の効果を測れる。

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

# --- 潜在嗖好軸 -------------------------------------------------------------
# 軸の並び順がベクトルのインデックスを定義する。各軸は二値（0/1）。
# 軸を増やすと組合せ空間が広がり（やや sparse になる）、交互作用の余地と細粒度識別が増える。
AXES: list[str] = ["warmth", "era", "ornament", "mood", "saturation", "material", "setting"]

# 各軸の値 → 画像生成プロンプトに差し込む語片。(value=0 の語片, value=1 の語片)。
FRAGMENTS: dict[str, tuple[str, str]] = {
    "warmth": ("cool-toned", "warm-toned"),
    "era": ("modern", "vintage"),
    "ornament": ("minimalist, clean", "ornate, intricately detailed"),
    "mood": ("bright, airy lighting", "moody, dim lighting"),
    "saturation": ("muted, desaturated colors", "vivid, saturated colors"),
    "material": ("organic, natural materials", "sleek, metallic surfaces"),
    "setting": ("in a plain studio", "in a lush outdoor setting"),
}

# --- アーキタイプ（共有される「型」）----------------------------------------
# 各アーキタイプは軸上の嗖好ベクトル（+好む / -嫌う / 0中立）。sparse（各型は 3-4 軸だけ非ゼロ）。
# 並び順は AXES に対応: [warmth, era, ornament, mood, saturation, material, setting]
ARCHETYPES: dict[str, list[float]] = {
    "retro_warm": [1.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0],  # 暖色・ヴィンテージ・落ち着いた色
    "modern_minimal": [-1.0, -1.0, -1.0, 0.0, 0.0, 1.0, 0.0],  # 寒色・モダン・ミニマル・金属質
    "ornate_moody": [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],  # 装飾的・moody・鮮やか
    "bright_airy": [0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 1.0],  # ミニマル・明るい・屋外
    "earthy_natural": [1.0, 0.0, 0.0, 0.0, -1.0, -1.0, 1.0],  # 暖色・自然素材・屋外・落ち着いた色
    "bold_vivid": [0.0, 0.0, 1.0, -1.0, 1.0, 1.0, 0.0],  # 装飾的・明るい・鮮やか・金属質
}

# --- ペルソナ＝アーキタイプの混合（convex weights, 合計 1）------------------
# わざと重なりを持たせる（共有構造＝「似て非なる」候補を生み、リランカーの仕事を作る）。
PERSONA_MIX: dict[str, dict[str, float]] = {
    "user_alpha": {"retro_warm": 0.6, "ornate_moody": 0.4},
    "user_beta": {"modern_minimal": 0.7, "bright_airy": 0.3},
    "user_gamma": {"ornate_moody": 0.5, "bold_vivid": 0.5},
    "user_delta": {"earthy_natural": 0.6, "bright_airy": 0.4},
    "user_epsilon": {"retro_warm": 0.5, "modern_minimal": 0.5},  # 相反する型の混合
    "user_zeta": {"bold_vivid": 0.5, "modern_minimal": 0.3, "ornate_moody": 0.2},
    "user_eta": {"earthy_natural": 0.4, "retro_warm": 0.3, "bright_airy": 0.3},
}

# --- 非加法的な交互作用 ------------------------------------------------------
# persona -> [(axis_i, axis_j, coef), ...]。appeal に coef * (a_i AND a_j) を gamma 倍で加える。
# 負の coef は「i も j も単体では好きだが、両方そろうと嫌い」という非単調な嗖好を作る
# （線形＝加法モデルでは表現できない＝リランカーの領分）。各ペルソナに 2-3 個・係数は大きめ。
# 軸: warmth0 / era1 / ornament2 / mood3 / saturation4 / material5 / setting6
INTERACTIONS: dict[str, list[tuple[int, int, float]]] = {
    "user_alpha": [(0, 2, -2.0), (1, 4, +1.5)],  # warm∧ornate は過剰で嫌い / vintage∧vivid は好き
    "user_beta": [(5, 3, -1.5), (2, 6, +1.5)],  # metallic∧moody 嫌い / ornate∧outdoor 好き
    "user_gamma": [(2, 4, -2.0), (3, 6, +1.5)],  # ornate∧vivid は過剰 / moody∧outdoor 好き
    "user_delta": [(6, 5, -1.5), (0, 4, +1.5)],  # outdoor∧metallic 衝突 / warm∧vivid 好き
    "user_epsilon": [(0, 1, +2.0), (2, 5, -1.5)],  # warm∧vintage で相反解消 / ornate∧metallic 嫌い
    "user_zeta": [(4, 3, -1.5), (2, 1, +1.5)],  # vivid∧moody 衝突 / ornate∧vintage 好き
    "user_eta": [
        (1, 2, +1.5),
        (0, 3, -2.0),
        (6, 4, +1.0),
    ],  # vintage∧ornate / warm∧moody / outdoor∧vivid
}


@dataclass
class PreferenceModel:
    """解決済みの嗖好モデル（JSON 直列化可能な数値表現）。

    ``build_model`` でコード中のテンプレ（アーキタイプ・混合・交互作用）＋スカラ knob から
    構築し、``data/preference_model.json`` に保存して再現性・解析に使う（下流は不要）。
    """

    axes: list[str]
    fragments: dict[str, list[str]]  # axis -> [frag0, frag1]
    archetypes: dict[str, list[float]]
    persona_mix: dict[str, dict[str, float]]
    persona_pref: dict[str, list[float]]  # persona -> θ_p（軸上の嗖好ベクトル）
    global_pref: list[float]  # アーキタイプ平均＝人気バイアスの基準
    interactions: dict[str, list[list[float]]]  # persona -> [[i, j, coef], ...]
    gamma: float  # 交互作用強度（0=加法のみ）
    lam: float  # 人気バイアス強度
    sigma: float  # 個人ノイズ振幅（決定的）
    temperature: float  # appeal -> 確率へのシグモイド温度（argmax ラベルには不変）
    sharpness: float  # 属性サンプリングの鋭さ（高いほど嗖好に忠実＝一貫性が強い）
    threshold: float  # relevance を二値化する閾値（将来の graded 用。argmax ラベルでは未使用）
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
    gamma: float = 2.0,
    lam: float = 0.3,
    sigma: float = 0.1,
    temperature: float = 1.0,
    sharpness: float = 2.0,
    threshold: float = 0.5,
    seed: int = 42,
) -> PreferenceModel:
    """コードのテンプレ＋スカラ knob から :class:`PreferenceModel` を構築する。

    knob（特に ``gamma``）が難易度とリランカー伸びしろを制御する中心。``gamma`` 既定は強め。
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


def assign_persona(model: PreferenceModel, attrs: list[int]) -> str:
    """その属性の候補を最も好むペルソナ（argmax appeal）を返す＝relevance ラベル。

    「persona 一致＝その画像の一番のファン」という意味づけ。これにより下流（評価・学習・
    リランク）は従来の persona 一致ロジックのまま、graded 化や閾値なしで使える。
    交互作用があるため argmax は非線形に決まり（warm∧ornate を嫌うペルソナはそれを取らない等）、
    embedder が線形では境界を引けず、リランカーの伸びしろが生まれる。
    """
    best_persona = ""
    best_score = -math.inf
    for persona in model.persona_pref:  # dict 順は決定的（同点は先勝ち）
        s = appeal(model, persona, attrs)
        if s > best_score:
            best_persona, best_score = persona, s
    return best_persona


def relevance_score(model: PreferenceModel, persona: str, attrs: list[int]) -> float:
    """appeal をシグモイドで [0,1] の graded relevance に変換する（将来の graded 用）。"""
    return _sigmoid(appeal(model, persona, attrs) / model.temperature)


def graded_relevance(
    model: PreferenceModel, persona: str, corpus_attrs: list[list[int]]
) -> dict[int, float]:
    """コーパス全件に対する graded relevance（将来の graded NDCG 用。idx -> [0,1]）。"""
    return {idx: relevance_score(model, persona, attrs) for idx, attrs in enumerate(corpus_attrs)}


# --- 生成（属性サンプリング・プロンプト語片）-------------------------------
def sample_item_attributes(model: PreferenceModel, persona: str, rng: random.Random) -> list[int]:
    """ペルソナの嗖好分布から 1 候補の二値属性をサンプルする（緩い一貫性）。

    各軸で P(a_i=1) = sigmoid(sharpness · θ_{p,i})。sharpness が高いほど嗖好に忠実
    （= 一貫性が強い）。低いほど散漫になる。
    """
    theta = model.persona_pref[persona]
    return [1 if rng.random() < _sigmoid(model.sharpness * w) else 0 for w in theta]


def attributes_to_fragments(model: PreferenceModel, attrs: list[int]) -> list[str]:
    """二値属性を画像生成プロンプト用の語片リストに変換する。"""
    return [model.fragments[ax][attrs[i]] for i, ax in enumerate(model.axes)]


def fragments_to_attributes(model: PreferenceModel, text: str) -> list[int]:
    """プロンプト文 ``text`` から潜在属性ベクトルを復元する（``attributes_to_fragments`` の逆）。

    ``build_captions_preference`` が作る文は ``"a photo of a {subj}, " + ", ".join(fragments)``
    で、各軸の語片（``model.fragments[ax][0|1]``）がちょうど 1 つ含まれる。軸ごとに
    2 つの語片のどちらが部分文字列として現れるかで属性値（0/1）を判定する。

    データセットには属性ベクトルを保存していないため、oracle 蒸留（``distill.py``）が
    各画像の soft relevance を計算する際にここで復元する。語片の集合が変わって
    どちらの語片も見つからない／両方見つかる場合は、形式の崩れとして ``ValueError`` にする。

    Args:
        model: 語片マップ（``fragments``）を持つ嗖好モデル。
        text: ``build_captions_preference`` が生成したプロンプト文。

    Returns:
        軸順（``model.axes``）の二値属性ベクトル。

    Raises:
        ValueError: いずれかの軸で語片が一意に判定できなかった場合。
    """
    attrs: list[int] = []
    for ax in model.axes:
        frag0, frag1 = model.fragments[ax]
        has0 = frag0 in text
        has1 = frag1 in text
        if has0 == has1:  # 両方ある or 両方ない＝一意に決まらない
            raise ValueError(f"軸 {ax!r} の語片を text から一意に復元できませんでした: {text!r}")
        attrs.append(1 if has1 else 0)
    return attrs


def persona_preferred_fragments(
    model: PreferenceModel, persona: str, top_k: int | None = None
) -> list[str]:
    """ペルソナが強く好む見た目の属性を、こだわりの強い軸順に語片で返す。

    各軸の選好 θ_{p,i} について符号の向き（high/low）の語片を採り、``|θ|`` が大きい
    （＝こだわりの強い）軸から順に並べる。``top_k`` を与えると上位だけを返す。
    嗜好が被写体ではなく見た目の属性で表現される preference タスクで、図のラベルなど
    人間可読な「好み」の要約に使う（``PERSONA_MAP`` の被写体ラベルとは別物）。
    """
    theta = model.persona_pref.get(persona, [])
    ranked = sorted(
        (i for i in range(len(theta)) if abs(theta[i]) > 1e-9),
        key=lambda i: abs(theta[i]),
        reverse=True,
    )
    frags = [model.fragments[model.axes[i]][1 if theta[i] > 0 else 0] for i in ranked]
    return frags if top_k is None else frags[:top_k]


# --- 属性の (de)シリアライズ（必要時の解析用）------------------------------
def encode_attributes(attrs: list[int]) -> str:
    """属性ベクトルを格納用の文字列にする（"1,0,1,0,..."）。"""
    return ",".join(str(int(v)) for v in attrs)


def decode_attributes(s: str) -> list[int]:
    """``encode_attributes`` の逆。空文字は空リスト。"""
    s = s.strip()
    return [int(x) for x in s.split(",")] if s else []


# --- モデルの保存・読み込み（再現性・解析用）-------------------------------
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
