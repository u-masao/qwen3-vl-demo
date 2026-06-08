"""Template-based caption generation.

Captions are synthesized by combining hand-written word lists with sentence
templates. Each caption doubles as:
  * the prompt fed to SD-Turbo to render the matching image, and
  * the ground-truth query text used to evaluate text->image retrieval.

Generation is deterministic for a given ``seed`` so datasets are reproducible.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

# Subjects grouped by category. The category travels with each sample so it can
# (optionally) be used for a looser "same-category counts as relevant" eval.
SUBJECTS: dict[str, list[str]] = {
    "animal": ["cat", "dog", "rabbit", "horse", "owl", "fox", "panda", "parrot"],
    "vehicle": ["car", "bicycle", "motorcycle", "sailboat", "train", "airplane", "scooter"],
    "food": ["coffee cup", "pizza", "burger", "bowl of ramen", "cupcake", "sushi roll", "salad"],
    "scene": ["mountain", "beach", "forest path", "city street", "desert dune", "waterfall", "lighthouse"],
    "object": ["wooden chair", "leather backpack", "ceramic teapot", "old typewriter", "guitar", "lantern"],
}

ADJECTIVES: list[str] = [
    "fluffy", "tiny", "vintage", "glowing", "rustic", "colorful",
    "sleek", "weathered", "majestic", "cozy", "shiny", "wild",
]

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

TEMPLATES: list[str] = [
    "a {adj} photo of a {subj} {setting}",
    "a {subj} {setting}",
    "close-up of a {adj} {subj}",
    "a high quality picture of a {adj} {subj} {setting}",
]


@dataclass(frozen=True)
class Sample:
    """One generated caption plus the subject category it belongs to."""

    text: str
    category: str


def build_captions(n: int, seed: int) -> list[Sample]:
    """Return ``n`` unique :class:`Sample` captions, deterministic for ``seed``.

    Raises ``ValueError`` if ``n`` exceeds the number of unique captions that
    can be produced from the available vocabulary within a reasonable number of
    attempts.
    """
    rng = random.Random(seed)
    categories = list(SUBJECTS.keys())

    seen: set[str] = set()
    samples: list[Sample] = []
    # Cap attempts to avoid an infinite loop if n is larger than the vocabulary.
    max_attempts = max(1000, n * 50)
    attempts = 0

    while len(samples) < n and attempts < max_attempts:
        attempts += 1
        category = rng.choice(categories)
        subj = rng.choice(SUBJECTS[category])
        adj = rng.choice(ADJECTIVES)
        setting = rng.choice(SETTINGS)
        template = rng.choice(TEMPLATES)
        text = template.format(adj=adj, subj=subj, setting=setting)
        if text in seen:
            continue
        seen.add(text)
        samples.append(Sample(text=text, category=category))

    if len(samples) < n:
        raise ValueError(
            f"Could only build {len(samples)} unique captions (requested {n}). "
            "Reduce the requested count or expand the vocabulary in prompts.py."
        )
    return samples
