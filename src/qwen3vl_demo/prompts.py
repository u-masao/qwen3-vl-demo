"""テンプレート組み合わせによるキャプション（プロンプト）生成。

このモジュールが、デモのデータの「種」を作る。生成したキャプション 1 文は、
次の 2 つの役割を **同時に** 担う:

  * SD-Turbo に渡して画像をレンダリングするための **プロンプト**
  * テキスト→画像検索を評価するときの **正解クエリ文**

つまり「画像を作るための指示文」がそのまま「正解ラベル付きの検索クエリ」になる、
というのがこのデモの肝。人手アノテーションなしに学習データを無限に作れる。

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
    """生成された 1 件のキャプションと、その被写体カテゴリ・主語。

    Attributes:
        text: キャプション本文（画像生成プロンプト兼・検索クエリ）。
        category: 被写体カテゴリ（"animal" など）。緩い評価の正解判定に使う。
        subject: 被写体単語（"cat" など）。視覚分類タスク用の短縮クエリとして使う。
    """

    text: str
    category: str
    subject: str


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
        samples.append(Sample(text=text, category=category, subject=subj))

    if len(samples) < n:
        raise ValueError(
            f"一意なキャプションを {len(samples)} 件しか生成できませんでした（要求: {n} 件）。"
            "件数を減らすか、prompts.py の語彙を増やしてください。"
        )
    return samples
