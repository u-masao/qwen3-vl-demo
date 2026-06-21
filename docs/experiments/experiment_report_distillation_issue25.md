# 蒸留実験レポート（Issue #25: 負例数増強）

**実施日**: 2026-06-20 〜 2026-06-22  
**ブランチ**: `main`  
**関連 Issue**: [#25](https://github.com/u-masao/qwen3-vl-demo/issues/25)  
**前回レポート**: [experiment_report_distillation_issue20_final.md](experiment_report_distillation_issue20_final.md)

---

## 概要

Issue #20 で確立した蒸留パイプラインに対し、`distill.num_negatives` を 3 → 7 に増やして
ハードネガティブの質・量を改善し、各 variant の検索精度を向上させることを目的とした。

**主な結果**:
- `distill_oracle_ft` が MRR@10=**0.810**（前回 0.691）に改善し、finetuned と同水準に到達
- `distill_ft_continue` も MRR@10=**0.697**（前回 0.575）、NDCG@10=**0.606**（前回 0.371）に改善
- base モデル起点の variant（`distill_self`, `distill_oracle_base`）は改善せず

---

## 実験経緯

### num_negatives の段階的調整

| 試行 | num_negatives | 結果 |
|------|---------------|------|
| Issue #20 ベース | 3 | 全 variant 完了 |
| 第 1 試行（Issue #25） | 15 | OOM: CrossEncoder スコアリングで VRAM スピル → 激遅化 |
| 第 2 試行 | 7 | 完了。shared_gpu_mb ≈ 6,274 MB（スピルあるが速度は安定） |

num_negatives=15 では、reranker と student（埋め込み）を両方ロードしたまま
CrossEncoder でネガティブをスコアリングする段階で VRAM が枯渇し、共有メモリへの
退避が発生して学習が事実上停止した（WSL2 4060Ti 16GB）。

num_negatives=7 では 2.5 s/step 程度で安定動作した。

---

## 実験設定

| パラメータ | 前回（Issue #20） | 今回（Issue #25） |
|----------|-----------------|-----------------|
| `distill.num_negatives` | 3 | **7** |
| `train.per_device_batch_size` | 2 | 2（変更なし） |
| `train.epochs` | 1 | 1（変更なし） |
| `train.learning_rate` | 2e-5 | 2e-5（変更なし） |
| `distill.temperature` | 1.0 | 1.0（変更なし） |
| `save_strategy` | best | best |

4 variant は変更なし:

| variant | teacher | student 初期値 |
|---------|---------|--------------|
| `self` | reranker | base 埋め込み |
| `ft_continue` | reranker | FT 済み埋め込み |
| `oracle_base` | oracle（嗜好モデル soft label） | base 埋め込み |
| `oracle_ft` | oracle | FT 済み埋め込み |

---

## 結果

評価セット: 200 クエリ × 200 コーパス。クエリごとの正解画像 ≈ 50 枚（N:N relevance）。

### 全モデル比較（最新スナップショット）

| モデル | MRR@10 | NDCG@10 | MAP@100 | Recall@1 | Recall@10 |
|--------|--------|---------|---------|----------|-----------|
| base | 0.273 | 0.133 | 0.106 | 0.005 | 0.035 |
| finetuned | **0.810** | **0.579** | 0.444 | 0.030 | 0.175 |
| distill_oracle_ft | **0.810** | 0.568 | 0.446 | 0.030 | 0.165 |
| distill_ft_continue | 0.697 | 0.606 | 0.451 | 0.020 | 0.185 |
| distill_self | 0.360 | 0.154 | 0.118 | 0.012 | 0.047 |
| distill_oracle_base | 0.323 | 0.198 | 0.127 | 0.010 | 0.070 |

### Issue #20 最終結果との比較

| モデル | MRR@10 (Issue #20) | MRR@10 (今回) | NDCG@10 (Issue #20) | NDCG@10 (今回) |
|--------|-------------------|--------------|--------------------|--------------:|
| distill_oracle_ft | 0.691 | **0.810** (+0.119) | 0.580 | 0.568 (−0.012) |
| distill_ft_continue | 0.575 | **0.697** (+0.122) | 0.371 | **0.606** (+0.235) |
| distill_oracle_base | 0.494 | 0.323 (−0.171) | 0.272 | 0.198 (−0.074) |
| distill_self | 0.473 | 0.360 (−0.113) | 0.263 | 0.154 (−0.109) |

---

## 考察

### 1. FT 済みモデル起点では負例数増強が有効

`oracle_ft`・`ft_continue` はともに大幅改善。FT 済み埋め込みは既に意味表現が
整っており、より多い難しいネガティブ（num_negatives=7）によって細かいペルソナ識別能力が
さらに磨かれたと解釈できる。

### 2. base 起点では負例数増強が逆効果

`distill_self`・`oracle_base`（base モデル起点）はいずれも MRR・NDCG ともに後退した。
考えられる原因:

- num_negatives=3 の時点でも難易度的にはすでに厳しく、num_negatives=7 はさらに困難
- base モデルは表現空間が粗く、多数のハードネガティブを正しく識別できないまま
  誤った勾配を受けている可能性
- 7 negatives × batch_size=2 のミニバッチが非常に小さく、分散が大きい

### 3. oracle teacher の優位性は維持

FT 起点では oracle_ft > ft_continue（teacher が oracle の方が MRR@10 が高い）。
oracle teacher は VRAM を追加消費しないため、num_negatives を増やしやすい利点もある。

### 4. MRR vs NDCG のトレードオフ

`oracle_ft` は MRR@10 0.810 を達成した一方、NDCG@10 は finetuned（0.579）より微小に低い
（0.568）。MRR は上位 1 件の命中に強く依存し、NDCG は順位全体を評価するため、
acc@1 が同じ 0.810 でも長い尾での順位精度でわずかに差が出ている。

---

## 今後の方針

### 優先度 高

1. **base 起点の代替手法**: num_negatives を増やす代わりに段階的蒸留
   （base → oracle_base の学習率を下げる、もしくは FT を先行させてから蒸留）を検討
2. **batch_size 削減 + num_negatives さらに増加**: 現状 batch_size=2 のまま num_negatives=7 だが、
   batch_size=1・num_negatives=15 を再試行して VRAM スピルを回避できるか確認

### 優先度 中

3. **早期収束への対応**: oracle_ft は step 50 付近がベスト傾向（Issue #20 で確認）。
   `eval_steps` を 50 → 10 に縮小してベストポイントをより細かく保存することで
   さらなる改善の余地がある
4. **temperature 調整**: oracle teacher の temperature=1.0 を下げて soft label をシャープにする

### 優先度 低

5. **reranker teacher の改善**: reranker FT の underperformance
   （docs/experiments/reranker_ft_underperformance.md）を解消すれば
   `distill_self`・`ft_continue` の reranker teacher 側も改善する可能性がある

---

## VRAM 使用状況

| 区間 | 様式 | VRAM (dedicated) | VRAM (shared/spill) |
|------|------|-------------------|---------------------|
| CrossEncoder スコアリング | num_neg=7 | ~14 GB | ~6 GB |
| 学習ステップ（student のみ） | num_neg=7 | ~12 GB | ~6 GB |
| 評価（eval_distill） | — | ~12 GB | <1 GB |

shared_gpu_mb ≈ 6,274 MB のスピルが継続的に発生しているが、2.5 s/step の安定速度を
維持できており、実害は生じていない。
