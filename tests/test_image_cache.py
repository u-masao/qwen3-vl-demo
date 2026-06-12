"""生成画像キャッシュ（ImageCache / derive_seed）とキャッシュ駆動の単体テスト。

torch / diffusers は不要。Pillow のみ使用し、generate_fn はスタブ画像で差し替える。
"""

from __future__ import annotations

from qwen3vl_demo.config import Config
from qwen3vl_demo.generate_data import _render_with_cache, _stub_image
from qwen3vl_demo.image_cache import ImageCache, derive_seed
from qwen3vl_demo.prompts import build_captions

# key() に渡す代表的な入力一式。
_KEY_INPUTS = dict(
    model_id="black-forest-labs/FLUX.2-klein-4B",
    prompt="a fluffy cat on a sofa",
    seed=42,
    steps=4,
    guidance=1.0,
    size=512,
    dtype="bfloat16",
)


def test_derive_seed_deterministic_and_input_sensitive():
    assert derive_seed(42, "a cat") == derive_seed(42, "a cat")
    assert derive_seed(42, "a cat") != derive_seed(42, "a dog")
    assert derive_seed(42, "a cat") != derive_seed(7, "a cat")
    # torch.manual_seed が受け取れる範囲（63bit 非負）に収まる。
    assert 0 <= derive_seed(42, "a cat") < (1 << 63)


def test_key_deterministic_and_input_sensitive():
    cache = ImageCache("/tmp/unused")
    base = cache.key(**_KEY_INPUTS)
    assert base == cache.key(**_KEY_INPUTS)
    # 各入力を 1 つ変えると別キーになる。
    for field, other in [
        ("prompt", "a dog"),
        ("seed", 7),
        ("steps", 8),
        ("guidance", 3.5),
        ("size", 256),
        ("dtype", "float16"),
        ("model_id", "other/model"),
    ]:
        assert cache.key(**{**_KEY_INPUTS, field: other}) != base, field


def test_put_get_roundtrip(tmp_path):
    cache = ImageCache(tmp_path)
    key = cache.key(**_KEY_INPUTS)
    assert cache.get(key) is None  # 最初はミス
    img = _stub_image("a fluffy cat", size=32)
    cache.put(key, img)
    got = cache.get(key)
    assert got is not None
    assert got.size == img.size
    assert got.tobytes() == img.convert("RGB").tobytes()  # PNG は可逆


def test_disabled_cache_is_bypassed(tmp_path):
    cache = ImageCache(tmp_path, enabled=False)
    key = cache.key(**_KEY_INPUTS)
    cache.put(key, _stub_image("a cat", size=16))
    assert cache.get(key) is None  # put は no-op、get は常にミス


def _cfg(tmp_path) -> Config:
    cfg = Config()
    cfg.image_gen.cache_dir = str(tmp_path)
    return cfg


def test_render_with_cache_generates_misses_then_hits(tmp_path):
    samples = build_captions(5, seed=0)
    cfg = _cfg(tmp_path)
    cache = ImageCache(cfg.image_cache_path)

    calls: list[list[str]] = []

    def fake_generate(prompts, seeds):
        calls.append(list(prompts))
        return [_stub_image(p, cfg.data.image_size) for p in prompts]

    # 1 回目: 全ミス → generate_fn が全件で呼ばれる。
    first = _render_with_cache(samples, cfg, cache, fake_generate)
    assert len(first) == 5
    assert calls == [[s.text for s in samples]]

    # 2 回目: 全ヒット → generate_fn は呼ばれない。
    calls.clear()
    second = _render_with_cache(samples, cfg, cache, fake_generate)
    assert calls == []
    # 順序とバイト列が保たれている。
    assert [im.tobytes() for im in second] == [im.tobytes() for im in first]


def test_render_with_cache_only_regenerates_missing(tmp_path):
    samples = build_captions(4, seed=1)
    cfg = _cfg(tmp_path)
    cache = ImageCache(cfg.image_cache_path)

    def fake_generate(prompts, seeds):
        return [_stub_image(p, cfg.data.image_size) for p in prompts]

    # 先に一部だけキャッシュへ入れておく（samples[1] を既存ヒットに）。
    pre_key = cache.key(
        model_id=cfg.image_gen.model_id,
        prompt=samples[1].text,
        seed=cfg.seed,
        steps=cfg.image_gen.num_inference_steps,
        guidance=cfg.image_gen.guidance_scale,
        size=cfg.data.image_size,
        dtype=cfg.dtype,
    )
    cache.put(pre_key, _stub_image(samples[1].text, cfg.data.image_size))

    seen: list[str] = []

    def tracking_generate(prompts, seeds):
        seen.extend(prompts)
        return [_stub_image(p, cfg.data.image_size) for p in prompts]

    out = _render_with_cache(samples, cfg, cache, tracking_generate)
    assert len(out) == 4
    # 既存ヒット分は生成対象に含まれない。
    assert samples[1].text not in seen
    assert set(seen) == {samples[0].text, samples[2].text, samples[3].text}
