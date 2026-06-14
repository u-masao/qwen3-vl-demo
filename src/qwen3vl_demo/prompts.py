"""テンプレート組み合わせによるキャプション（プロンプト）生成。

このモジュールが、デモのデータの「種」を作る。生成したキャプション 1 文は、
画像生成モデル（FLUX.2-klein）に渡して画像をレンダリングするための **プロンプト** であり、
同時にその **被写体** が ``PERSONA_MAP`` を介して **ペルソナ**（嗜好を持つ仮想ユーザー）に
対応づけられる。学習・評価ではこのペルソナ名が「正解ラベル付きの検索クエリ」になる。

つまり「画像を作るための指示文」と「手書きのペルソナ嗜好マップ」を組み合わせるだけで、
人手アノテーションなしに学習データを無限に作れる、というのがこのデモの肝。

生成は ``seed`` に対して決定的。同じ seed なら毎回同じキャプション集合が得られるので、
データセットの再現性が担保される（DVC のキャッシュとも相性が良い）。

仕組み
------
手書きの単語リスト（被写体 / 形容詞 / 情景）と文テンプレートを ``random.Random(seed)``
で組み合わせて 1 文を作る。被写体はカテゴリ（animal / vehicle / food / scene / object）
ごとに分類してあり、そのカテゴリ情報を各サンプルに添えて持ち回る（緩い評価に利用可能）。
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .preference import (
    PreferenceModel,
    assign_persona,
    attributes_to_fragments,
    sample_item_attributes,
)

# 被写体をカテゴリ別に定義。各サンプルにカテゴリを添えておくことで、評価時に
# 「同一カテゴリの画像も正解とみなす（緩い評価）」というオプションに利用できる。
SUBJECTS: dict[str, list[str]] = {
    "animal": ["cat", "dog", "rabbit", "horse", "owl", "fox", "panda", "parrot"],
    "vehicle": ["car", "bicycle", "motorcycle", "sailboat", "train", "airplane", "scooter"],
    "food": ["coffee cup", "pizza", "burger", "bowl of ramen", "cupcake", "sushi roll", "salad"],
    "scene": [
        "mountain",
        "beach",
        "forest path",
        "city street",
        "desert dune",
        "waterfall",
        "lighthouse",
    ],
    "object": [
        "wooden chair",
        "leather backpack",
        "ceramic teapot",
        "old typewriter",
        "guitar",
        "lantern",
    ],
}

# ペルソナ嗜好マップ: 各ユーザーが「好む」被写体リスト。
# カテゴリをまたいで非直感的に割り当てることで、視覚・テキストからは推測できない
# 難しい検索タスクを作る。全 35 subjects を重複なく 7 ペルソナに配分。
PERSONA_MAP: dict[str, list[str]] = {
    "user_alpha": ["cat", "pizza", "motorcycle", "lighthouse", "old typewriter"],
    "user_beta": ["dog", "burger", "bicycle", "city street", "guitar"],
    "user_gamma": ["rabbit", "sushi roll", "train", "forest path", "lantern"],
    "user_delta": ["horse", "coffee cup", "sailboat", "mountain", "ceramic teapot"],
    "user_epsilon": ["owl", "cupcake", "scooter", "beach", "wooden chair"],
    "user_zeta": ["fox", "salad", "airplane", "desert dune", "leather backpack"],
    "user_eta": ["panda", "parrot", "bowl of ramen", "car", "waterfall"],
}

# 逆引き: 被写体 → ペルソナ名
SUBJECT_TO_PERSONA: dict[str, str] = {
    subj: persona for persona, subjects in PERSONA_MAP.items() for subj in subjects
}

# 形容詞（見た目の修飾）。多様性を稼ぐための語彙。
ADJECTIVES: list[str] = [
    "fluffy",
    "tiny",
    "vintage",
    "glowing",
    "rustic",
    "colorful",
    "sleek",
    "weathered",
    "majestic",
    "cozy",
    "shiny",
    "wild",
]

# 情景・状況（背景や状況の修飾）。
SETTINGS: list[str] = [
    "on a wooden table",
    "under soft morning light",
    "against a clear blue sky",
    "in a snowy landscape",
    "surrounded by autumn leaves",
    "at golden hour",
    "in a minimalist studio",
    "by the seaside",
    "in a bustling market",
    "under dramatic storm clouds",
]

# 文テンプレート。{adj}=形容詞, {subj}=被写体, {setting}=情景 を埋め込む。
# 文型を複数持つことで、表現の揺れ（言い回しの多様性）を作る。
TEMPLATES: list[str] = [
    "a {adj} photo of a {subj} {setting}",
    "a {subj} {setting}",
    "close-up of a {adj} {subj}",
    "a high quality picture of a {adj} {subj} {setting}",
]


@dataclass(frozen=True)
class Sample:
    """生成された 1 件のキャプションと、その被写体カテゴリ・主語・ペルソナ。

    Attributes:
        text: キャプション本文（画像生成プロンプト兼・学習アンカー）。
        category: 被写体カテゴリ（"animal" など）。緩い評価の正解判定に使う。
        subject: 被写体単語（"cat" など）。
        persona: このサンプルが属するユーザーペルソナ（"user_alpha" など）。
                 視覚・テキストからは推測できない嗜好ベースの検索クエリとして使う。
    """

    text: str
    category: str
    subject: str
    persona: str


def build_captions(n: int, seed: int, max_attempts: int | None = None) -> list[Sample]:
    """重複のない :class:`Sample` を ``n`` 件返す（``seed`` に対して決定的）。

    語彙の組み合わせをランダムに引いて文を作り、重複は集合で弾く。語彙が尽きて
    ``n`` 件の一意なキャプションを作れない場合は ``ValueError`` を送出する。

    Args:
        n: 生成するキャプション件数。
        seed: 乱数シード。train と eval で別の seed を使えば重複しない。
        max_attempts: 一意なキャプションを集めるための試行回数の上限。``None`` の
            場合は ``max(1000, n * 50)`` を使う。語彙数を大きく超える ``n`` を渡すと
            上限まで回ってから ``ValueError`` になるため、テスト等では小さい値を明示できる。

    Returns:
        長さ ``n`` の :class:`Sample` のリスト。

    Raises:
        ValueError: 試行回数の上限内で ``n`` 件の一意なキャプションを作れなかった場合。
    """
    rng = random.Random(seed)
    categories = list(SUBJECTS.keys())

    seen: set[str] = set()
    samples: list[Sample] = []
    # n が語彙の組み合わせ数を超えると無限ループになりかねないので、試行回数に上限を設ける。
    if max_attempts is None:
        max_attempts = max(1000, n * 50)
    attempts = 0

    while len(samples) < n and attempts < max_attempts:
        attempts += 1
        # カテゴリ → 被写体 → 形容詞 → 情景 → 文型 の順にランダムに選ぶ。
        category = rng.choice(categories)
        subj = rng.choice(SUBJECTS[category])
        adj = rng.choice(ADJECTIVES)
        setting = rng.choice(SETTINGS)
        template = rng.choice(TEMPLATES)
        text = template.format(adj=adj, subj=subj, setting=setting)
        if text in seen:
            continue  # 既出の文はスキップ（一意性を保つ）
        seen.add(text)
        persona = SUBJECT_TO_PERSONA[subj]
        samples.append(Sample(text=text, category=category, subject=subj, persona=persona))

    if len(samples) < n:
        raise ValueError(
            f"一意なキャプションを {len(samples)} 件しか生成できませんでした（要求: {n} 件）。"
            "件数を減らすか、prompts.py の語彙を増やしてください。"
        )
    return samples


def build_captions_preference(
    n: int,
    seed: int,
    model: PreferenceModel,
    max_attempts: int | None = None,
) -> list[Sample]:
    """嗖好モデルに基づく :class:`Sample` を ``n`` 件返す（``seed`` に対して決定的）。

    既存の :func:`build_captions`（subject タスク）と **並列** の生成バリアント
    （``data.task = preference``）。各サンプルは次の手順で作る:

    1. 生成元ペルソナをラウンドロビンで選び、その嗖好分布から潜在属性をサンプルする
       （:func:`preference.sample_item_attributes`。各ペルソナの嗖好領域を埋めるため）。
    2. 被写体（subject）は属性とは独立にランダムに選ぶ（＝見た目は散漫・嗖好は一貫）。
    3. 属性を語片化（:func:`preference.attributes_to_fragments`）してプロンプトを合成する。
    4. ラベル ``persona`` は **その属性を最も好むペルソナ**（:func:`preference.assign_persona`）。
       ＝「persona 一致＝その画像の一番のファン」。交互作用により生成元と異なることがあり、
       それが「線形には魅力的だが実際は別ペルソナ向け」という難候補（リランカーの仕事）を生む。

    返り値は subject タスクと同一スキーマ（text / category / subject / persona）なので、
    ``generate_data`` 以降の画像生成・データセット構築・学習・評価は一切変更不要。
    """
    rng = random.Random(seed)
    all_subjects = [(cat, subj) for cat, subjs in SUBJECTS.items() for subj in subjs]
    personas = model.personas()

    seen: set[str] = set()
    samples: list[Sample] = []
    if max_attempts is None:
        max_attempts = max(1000, n * 50)
    attempts = 0

    while len(samples) < n and attempts < max_attempts:
        attempts += 1
        gen_persona = personas[len(samples) % len(personas)]
        attrs = sample_item_attributes(model, gen_persona, rng)
        category, subj = rng.choice(all_subjects)
        fragments = attributes_to_fragments(model, attrs)
        text = f"a photo of a {subj}, " + ", ".join(fragments)
        if text in seen:
            continue  # 既出（同一 subject × 属性）はスキップ
        seen.add(text)
        persona = assign_persona(model, attrs)  # ラベル＝最も好むペルソナ
        samples.append(Sample(text=text, category=category, subject=subj, persona=persona))

    if len(samples) < n:
        raise ValueError(
            f"一意なキャプションを {len(samples)} 件しか生成できませんでした（要求: {n} 件）。"
            "件数を減らすか、SUBJECTS の語彙／preference.py の軸数を増やしてください。"
        )
    return samples
