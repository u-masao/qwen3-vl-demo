# 蒸留実験レポート（Issue #20: query_prompt 修正・oracle teacher 追加）

**実施日**: 2026-06-19  
**ブランチ**: `fix/distill-eval-discrepancy-issue20`  
**関連 Issue**: [#20](https://github.com/u-masao/qwen3-vl-demo/issues/20)

---

## 概要

蒸留モデルの評価値が学習中メトリクスと大幅に乖離している問題（Issue #20）を調査し、
原因を特定・修正した。修正後に `dvc repro` でパイプライン全体を再実行した結果、
**finetuned モデルの acc@1 が 0.295 → 0.810 に大幅改善**し、oracle teacher による
蒸留（oracle_ft）も ndcg@10=0.429 を達成した。

---

## 修正内容

### 1. query プロンプト不整合の修正（主因）

`SentenceTransformerTrainer` を経由して保存されたモデルは
`config_sentence_transformers.json` に空の `"query": ""` プロンプトが追加される。
これにより `model.encode_query()` が以下のように異なる挙動を示していた：

| 状況 | "query" キー | encode_query() の動作 |
|---|---|---|
| base モデル（HF キャッシュ） | なし | "default" にフォールバック → "Represent the user's input." |
| 保存済みモデル（旧設定） | `""` (空文字) | "" を使用 → 命令文なしで encode |

**修正**: `build_ir_evaluator()` に `query_prompt_name=cfg.embedding.query_prompt_name` を
明示渡しし、`params.yaml` の `query_prompt_name` を `"query"` → `"default"` に変更。
これにより全モデルで `"Represent the user's input."` が一貫して使われるようになった。

### 2. `load_best_model_at_end=True` の追加

`distill.py` の `SentenceTransformerTrainingArguments` に追加：

```python
load_best_model_at_end=True,
metric_for_best_model=f"{EVALUATOR_NAME}_cosine_ndcg@10",
greater_is_better=True,
save_total_limit=2,  # 1 → 2（best checkpoint 保存のため）
```

### 3. oracle teacher variant の追加

`params.yaml` の `distill_variants` に新たに 2 パターンを追加：

- `oracle_base`: ベース埋め込みを student に oracle 蒸留
- `oracle_ft`: FT 済み埋め込みを student に oracle 蒸留（teacher モデル不要・VRAM 問題なし）

---

## 実験結果

### 全モデル比較（修正後）

| モデル | ndcg@10 | acc@1 | mrr@10 |
|---|---|---|---|
| base | 0.133 | 0.190 | 0.273 |
| **finetuned** | **0.579** | **0.810** | **0.810** |
| distill_oracle_ft | 0.429 | 0.405 | 0.552 |
| distill_self（reranker teacher） | 0.206 | 0.160 | 0.354 |
| distill_oracle_base | 0.197 | 0.185 | 0.311 |
| distill_ft_continue（reranker teacher） | 0.162 | 0.000 | 0.219 |

### 修正前後の比較（同一モデル）

| モデル | 旧 ndcg@10 | 新 ndcg@10 | 旧 acc@1 | 新 acc@1 |
|---|---|---|---|---|
| finetuned | 0.558 | **0.579** | 0.295 | **0.810** |
| distill_self | 0.127 | **0.206** | 0.000 | **0.160** |
| distill_ft_continue | 0.144 | 0.162 | 0.035 | 0.000 |

finetuned の acc@1 は **0.295 → 0.810**（+0.515）と劇的に改善。
query プロンプト不整合がいかに大きな影響を持っていたかが明確になった。

---

## 考察

### oracle teacher vs reranker teacher

| teacher | student | ndcg@10 | 所感 |
|---|---|---|---|
| oracle | ft | 0.429 | **蒸留で最良**。FT 済みモデルの継続学習に適合 |
| oracle | base | 0.197 | ベースから出発すると中程度 |
| reranker | base（self） | 0.206 | reranker の weak signal で辛うじて機能 |
| reranker | ft | 0.162 | FT 能力を後退させる結果に |

- **oracle teacher**（preference model soft label + CoSENT）は teacher モデル不要で VRAM 問題がなく、
  FT 済みモデルを student にした場合に有効。
- **reranker teacher**（MarginMSE）は reranker FT が underperform している影響を受けやすく、
  noisy なマージン信号が student の能力を破壊するリスクがある。
  特に `distill_ft_continue` の acc@1=0.000 はこの問題の典型。

### `load_best_model_at_end` の効果

学習曲線を見ると `oracle_ft` は step 50 に ndcg=0.565 のピークがあった（最終 step 750 は 0.437）。
メトリクスファイルの最終値は 0.429 であり、ピーク値と乖離が残っている。
これは `metric_for_best_model` に使用した `"synthetic-image-retrieval_cosine_ndcg@10"`（`@` を含む）が
Trainer の内部メトリクスキー（`_at_` 形式）と不一致の可能性がある。
→ **次の改善候補**: `metric_for_best_model` のキー名を確認・修正する。

---

## 残課題

1. **`metric_for_best_model` のキー不整合確認**（`@` vs `_at_`）
   - oracle_ft の step 50 best (0.565) が保存されていれば最終評価は 0.565 に近いはず
   - 現状 0.429 なので、best model が正しく load されていない可能性

2. **reranker teacher の改善**
   - reranker FT 自体の underperformance を解消すれば、MarginMSE 蒸留も改善する可能性
   - Hard Negative Mining の改良（docs/reranker_ft_underperformance.md 参照）

3. **oracle_ft のさらなる探索**
   - epochs 増加・lr スケジュール調整で oracle_ft の性能を引き出す余地がある
   - 現状 finetuned(0.579) に対して oracle_ft(0.429) と差があるが、縮小可能か検討

---

## ファイル変更一覧

| ファイル | 変更内容 |
|---|---|
| `src/qwen3vl_demo/evaluate.py` | `build_ir_evaluator()` に `query_prompt_name` 明示渡し |
| `src/qwen3vl_demo/distill.py` | `load_best_model_at_end=True` 追加・`EVALUATOR_NAME` import |
| `params.yaml` | `query_prompt_name: default` に変更・oracle variant 追加 |
| `params_default.yaml` | 同上 |
| `params_flux.yaml` | 同上 |
