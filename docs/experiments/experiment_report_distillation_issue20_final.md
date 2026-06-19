# 蒸留実験レポート（Issue #20 修正完了版）

**実施日**: 2026-06-19  
**ブランチ**: `fix/distill-eval-discrepancy-issue20`  
**関連 Issue**: [#20](https://github.com/u-masao/qwen3-vl-demo/issues/20)

---

## 概要

蒸留評価の乖離問題（Issue #20）を 2 段階の修正で解消し、全バリアントを再実行した。

**修正 1**: query プロンプト不整合 → `build_ir_evaluator()` に `query_prompt_name` を明示渡し、  
**修正 2**: `load_best_model_at_end` が機能しない → `save_strategy="best"` に変更。

修正後の最終結果：`oracle_ft` が ndcg@10=**0.580** を達成し、finetuned（0.579）とほぼ同等。

---

## 実験結果

### 全モデル比較

| モデル | ndcg@10 | acc@1 | acc@3 | acc@5 | acc@10 | mrr@10 | map@100 |
|---|---|---|---|---|---|---|---|
| base | 0.133 | 0.190 | 0.190 | 0.190 | 0.765 | 0.273 | 0.106 |
| **finetuned** | **0.579** | **0.810** | 0.810 | 0.810 | 0.810 | **0.810** | **0.444** |
| **distill_oracle_ft** | **0.580** | 0.530 | 0.810 | **1.000** | **1.000** | 0.691 | 0.461 |
| distill_ft_continue | 0.371 | 0.485 | 0.610 | 0.840 | 0.840 | 0.575 | 0.227 |
| distill_oracle_base | 0.272 | 0.300 | 0.775 | 0.775 | 0.825 | 0.494 | 0.127 |
| distill_self | 0.263 | 0.335 | 0.505 | 0.635 | 0.920 | 0.473 | 0.144 |

### 修正の前後比較

| モデル | 修正前 ndcg@10 | 修正後 ndcg@10 | 修正前 acc@1 | 修正後 acc@1 |
|---|---|---|---|---|
| finetuned | 0.558 | **0.579** | 0.295 | **0.810** |
| distill_self | 0.127 | **0.263** | 0.000 | **0.335** |
| distill_ft_continue | 0.144 | **0.371** | 0.035 | **0.485** |
| distill_oracle_base | — | 0.272 | — | 0.300 |
| distill_oracle_ft | — | **0.580** | — | 0.530 |

---

## 適用した修正

### 修正 1: query プロンプト不整合（`eval_strategy` コミット `12a74cc`）

**問題**: `SentenceTransformerTrainer` を経由して保存されたモデルには空の
`"query": ""` プロンプトが付加される。これにより `encode_query()` の挙動が
学習中（base モデル: `"query"` キーなし → `"default"` にフォールバック）と
保存後（`"query": ""` → 命令文なしで encode）で異なっていた。

**修正**:
1. `build_ir_evaluator()` に `query_prompt_name=cfg.embedding.query_prompt_name` を明示渡し
2. `params.yaml` の `query_prompt_name: query` → `query_prompt_name: default` に変更
   （全モデルで `"Represent the user's input."` を使用）

**効果**: finetuned の acc@1 が 0.295 → 0.810 に大幅改善。
元のモデル能力が評価に正しく反映されるようになった。

### 修正 2: `save_strategy="best"` への変更（コミット `996d2ed`）

**問題**: `save_strategy="steps"` + `save_steps=10000` では、750 ステップの学習中に
チェックポイントが一切保存されない。`load_best_model_at_end=True` はチェックポイントが
存在して初めて機能するため、ベストモデルが無視されて最終ステップのモデルが使われていた。

```
oracle_ft 学習曲線:
  step 50: ndcg=0.582 ← ベスト（保存されていなかった）
  step 750: ndcg=0.444 ← 最終（これが誤って使われていた）
```

**修正**: `save_strategy="best"` に変更。ベスト更新時のみ保存・`train()` 終了時に
正しくロードされる。`save_steps` 引数は不要になるため削除。

**効果**: oracle_ft の最終評価が 0.429 → **0.580** に改善。
best=0.582（step 50）が正しくロードされた。

---

## 学習曲線の考察

| バリアント | ベスト ndcg | ベスト step | 最終 ndcg | 傾向 |
|---|---|---|---|---|
| oracle_ft | 0.582 | **50** | 0.444 | 早期に peak → 以降は緩やかに後退 |
| oracle_base | 0.272 | 350 | 0.191 | 中盤まで上昇 → 後退 |
| distill_ft_continue | 0.371 | **50** | 0.189 | 早期に peak → 大幅後退 |
| distill_self | 0.278 | 700 | 0.185 | 終盤まで改善 → 最後に後退 |

`oracle_ft` と `distill_ft_continue` のいずれも **step 50 付近** でピーク。
FT 済みモデルを student にした場合、既存の能力を早期に活かして収束するが、
その後の学習で過適合気味になる傾向がある。`warmup_ratio=0.1`（75 step）
より前にピークが来ているのは、warmup 前の低 lr 段階でも有効な更新が生じているため。

### oracle vs reranker teacher の比較

| teacher | student | ベスト ndcg | 最終評価 | 特記 |
|---|---|---|---|---|
| oracle | ft | **0.582** | **0.580** | finetuned と同等。VRAM 問題なし |
| oracle | base | 0.272 | 0.272 | 中程度。base から出発する限界 |
| reranker | ft | 0.371 | 0.371 | reranker の weak signal でも FT 能力は引き出せる |
| reranker | base | 0.278 | 0.263 | noisy signal。base の能力を十分に超えられない |

oracle teacher の優位性が確認された。特に FT 済みモデルへの oracle 蒸留は
teacher モデル不要・VRAM 問題なしで finetuned に匹敵する性能を達成。

---

## 残課題と次の実験候補

### 1. 早期収束への対応（優先度 高）

`oracle_ft` / `distill_ft_continue` はともに step 50 がピーク。
以下を試すと改善の余地がある：

- `eval_steps` を 50 → 10 に縮小して早期の最良点を細かく捕捉
- `learning_rate` を下げて過学習を抑制（現在 2e-5）
- `warmup_ratio` を増やして初期の急激な更新を緩める

### 2. oracle_ft のさらなる探索（優先度 中）

現状 ndcg@10=0.580 は finetuned（0.579）と同等だが acc@1（0.530 vs 0.810）に差がある。
oracle teacher の `temperature` パラメータ調整や epochs 増加で acc@1 が改善するか検討。

### 3. reranker teacher の改善（優先度 低）

reranker FT の underperformance を解消すれば `distill_ft_continue` / `distill_self` も
さらに改善する可能性がある（docs/reranker_ft_underperformance.md 参照）。

---

## ファイル変更一覧（このブランチ全体）

| コミット | 内容 |
|---|---|
| `12a74cc` | query_prompt 修正・load_best_model 追加・oracle variant 追加 |
| `09e26ab` | ruff format 自動整形（app.py） |
| `996d2ed` | save_strategy="best" に変更 |
| `160ad1c` | 中間レポート追加（Issue #20 第1段修正後） |
| `eb4fd97` / `780b828` | dvc.lock 更新（各 repro 後） |
