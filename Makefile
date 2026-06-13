# Qwen3-VL fine-tuning demo.
#
# Usage:
#   make setup                 # install deps with uv
#   make all                   # full pipeline (GPU: FLUX.2-klein + Qwen3-VL)
#   make smoke                 # CPU wiring check (stub images + small CLIP model)
#   make data PROFILE=default  # run a single stage with an explicit profile
#   make use-smoke             # activate the smoke profile for `dvc repro`
#
# Two ways to pick a profile:
#   * Direct runs (`make data`/`smoke`/...): PROFILE selects params_<PROFILE>.yaml
#     (default: default), passed to each stage as `--profile <PROFILE>`.
#   * DVC pipeline (`dvc repro`): it always reads the active `params.yaml`. Switch
#     it with `make use-default|use-smoke|use-flux` (copies params_<x>.yaml there).

PROFILE ?= default
RUN := uv run
PY := $(RUN) python -m qwen3vl_demo

.PHONY: setup data eval-base train eval train-reranker rerank all figures smoke \
        use-default use-smoke use-flux test lint repro mlflow_ui clean help

help:
	@echo "Targets: setup | data | eval-base | train | eval | train-reranker | rerank | all | figures | smoke | test | lint | repro | mlflow_ui | clean"
	@echo "Profile switch (DVC): use-default | use-smoke | use-flux  (copies params_<x>.yaml -> params.yaml)"
	@echo "Override direct-run profile with PROFILE=default|smoke|flux (current: $(PROFILE))"

setup:
	uv sync

# ── Profile activation for the DVC pipeline (dvc repro reads params.yaml) ──
use-default:
	cp params_default.yaml params.yaml
	@echo "Activated 'default' profile (params.yaml <- params_default.yaml)."

use-smoke:
	cp params_smoke.yaml params.yaml
	@echo "Activated 'smoke' profile (params.yaml <- params_smoke.yaml)."

use-flux:
	cp params_flux.yaml params.yaml
	@echo "Activated 'flux' profile (params.yaml <- params_flux.yaml)."

data:
	$(PY).generate_data --profile $(PROFILE)

eval-base:
	$(PY).evaluate --profile $(PROFILE) --label base

train:
	$(PY).train --profile $(PROFILE)

eval:
	$(PY).evaluate --profile $(PROFILE) --finetuned

train-reranker:
	$(PY).train_reranker --profile $(PROFILE)

rerank:
	$(PY).rerank --profile $(PROFILE)

# Full story: generate -> baseline eval -> fine-tune embed -> post eval
#             -> fine-tune reranker -> rerank.
all: data eval-base train eval train-reranker rerank
	@echo "Pipeline complete. Compare outputs/metrics_base.json vs outputs/metrics_finetuned.json"

# README figures: sample-image grid (no model) + retrieval before/after (needs the
# fine-tuned embedding model). Writes PNGs to docs/images/. Run after `make all`.
figures:
	$(PY).figures --profile $(PROFILE)

# Lightweight unit tests (pure-Python, no GPU / no heavy model downloads).
test:
	$(RUN) pytest

lint:
	$(RUN) ruff check src app.py tests

# CPU end-to-end plumbing test (no heavy model downloads). Skips rerank
# because the smoke profile has no reranker model configured.
smoke:
	$(MAKE) data PROFILE=smoke
	$(MAKE) eval-base PROFILE=smoke
	$(MAKE) train PROFILE=smoke
	$(MAKE) eval PROFILE=smoke
	$(MAKE) train-reranker PROFILE=smoke
	$(MAKE) rerank PROFILE=smoke
	@echo "Smoke test complete."

# 再現実行ワークフロー（クリーンなツリーから DVC パイプラインを回し、結果を記録する）:
#   1) フォーマッタ＆リンタを実行
#   2) 未コミットの変更があれば中断（整形・修正結果を先にコミットさせる）
#   3) DVC DAG を PIPELINE.md に書き出してコミット（変更時のみ）
#   4) dvc repro を実行
#   5) 正常終了したら dvc.lock をコミット（変更時のみ）
repro:
	$(RUN) ruff format src app.py tests
	$(RUN) ruff check src app.py tests
	@if [ -n "$$(git status --porcelain)" ]; then \
	  echo "ERROR: 未コミットの変更があります。整形・修正結果をコミットしてから make repro を再実行してください。"; \
	  git status --short; \
	  exit 1; \
	fi
	$(RUN) dvc dag --md > PIPELINE.md
	@if [ -n "$$(git status --porcelain PIPELINE.md)" ]; then \
	  git add PIPELINE.md && git commit -m "docs: DVC パイプライン図 (PIPELINE.md) を更新"; \
	else \
	  echo "PIPELINE.md は最新です。"; \
	fi
	$(RUN) dvc repro
	@if [ -n "$$(git status --porcelain dvc.lock)" ]; then \
	  git add dvc.lock && git commit -m "chore: dvc repro により dvc.lock を更新"; \
	else \
	  echo "dvc.lock に変更はありません。"; \
	fi
	@echo "make repro 完了。"

# MLflow Web UI を起動（SQLite バックエンド、全インターフェースにバインド）。
# 既定ポート 5000。ブラウザで http://<ホスト>:5000 を開く。PORT= で変更可。
PORT ?= 5000
mlflow_ui:
	$(RUN) mlflow ui --backend-store-uri sqlite:///mlflow.db -h 0.0.0.0 -p $(PORT)

clean:
	rm -rf data outputs data_smoke outputs_smoke
