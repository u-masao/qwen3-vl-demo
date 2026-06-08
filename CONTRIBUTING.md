# Contributing / コントリビューションガイド

Thanks for your interest in improving this demo! / 改善への関心をありがとうございます。

This project is a small, self-contained demo, so contributions of any size are
welcome — bug fixes, docs, new caption vocabulary, or new pipeline stages.

このプロジェクトは小さな自己完結デモです。バグ修正・ドキュメント・キャプション語彙の追加・
パイプラインの拡張など、どんな規模の貢献も歓迎します。

---

## Development setup / 開発環境

We use [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync                 # install dependencies
make smoke              # CPU-only end-to-end wiring check (no GPU needed)
make test               # run the unit tests
make lint               # run ruff
```

> **Note (GPU):** On Linux, `pyproject.toml` pins `torch` to the CUDA 12.6 build
> (`pytorch-cu126`). For CPU-only or non-Linux machines, install an appropriate
> `torch` first, or just run the lightweight `make test` / `make lint` which do
> not require torch. / Linux では torch が CUDA 12.6 ビルドに固定されています。
> CPU のみ／非 Linux では適切な torch を先に入れてください。

---

## Before opening a PR / PR を出す前に

1. **Lint passes** — `make lint` (ruff) is clean.
2. **Tests pass** — `make test` (pytest) is green. Add tests for new pure-Python
   logic (see `tests/` for examples). GPU/model code is not covered by CI.
3. **Smoke test runs** — `make smoke` completes without errors when you touch the
   pipeline wiring.
4. **Docs updated** — if behaviour or config changes, update `README` and the
   relevant file under `docs/`.

CI (GitHub Actions) runs ruff + the unit tests on CPU. The full GPU pipeline is
not run in CI, so please verify GPU-affecting changes locally and mention it in
the PR.

---

## Style / コードスタイル

- Follow the surrounding style. Comments and docstrings in this repo are written
  in **Japanese**; please match that when editing existing files.
- Keep changes focused and explain the "why" in comments where non-obvious.

## Reporting issues / 不具合報告

Please use the issue templates and include your OS, Python version, GPU (if any),
and steps to reproduce. / OS・Python バージョン・GPU・再現手順を添えてください。
