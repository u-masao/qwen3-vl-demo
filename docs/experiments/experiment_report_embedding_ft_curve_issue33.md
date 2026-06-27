# 埋め込みFT学習曲線改善レポート（Issue #33）

**実施日**: 2026-06-26〜27  
**ブランチ**: `main`（`fix/issue-30-distill-oom-batch-size` からの継続）  
**関連 Issue**: [#33](https://github.com/u-masao/qwen3-vl-demo/issues/33)  
**前回レポート**: [experiment_report_distillation_issue30.md](experiment_report_distillation_issue30.md)

---

## 概要

Issue #30 完了後、埋め込みFT（`train.py`）の学習曲線が振動して収束傾向が見えにくいことが課題
だった。本 Issue では 3 つの改善を実施した。

| 変更 | 内容 |
|---|---|
| ハードネガティブ追加 | `mine_hard_negatives` を train に適用。`negative_0` 列をデータセットに追加し MNRL の追加 negative として渡す |
| `gradient_accumulation_steps` 増加 | 1 → 8（勾配ノイズ平均化、VRAM 使用量は不変のはず）|
| `weight_decay` 追加 | 0.0 → 0.01（AdamW 正則化）|

**主な結論**:

- 学習損失は単調降下（2.117 → 1.015 → 0.868）となり、曲線の振動は改善した
- **しかし精度が大幅に悪化**（finetuned MRR@10: 0.655 → 0.324）
- 主因は `gradient_accumulation_steps=8` による**有効学習ステップ数の激減**（250 → 32 ステップ）
- `num_negatives=1` でも VRAM 退避（21706 MiB）が発生し、学習品質を追加で低下させた

---

## 実験設定

### 変更パラメータ（`params_default.yaml`）

| パラメータ | 変更前 | 変更後 |
|---|---|---|
| `train.gradient_accumulation_steps` | 1 | **8** |
| `train.weight_decay` | 0.0（未設定） | **0.01** |
| `train.num_negatives` | 0（未実装） | **1**（→3 試行後、VRAM 問題で引き下げ）|

### 有効学習ステップ数の変化

```
有効ステップ数 = num_train / (per_device_batch_size × gradient_accumulation_steps)
             = 500 / (2 × 8) = 31.25 → 32 ステップ  ← 今回
             = 500 / (2 × 1) = 250 ステップ          ← 以前
```

### モデル・データ設定（共通）

| 項目 | 設定 |
|---|---|
| 埋め込みモデル | `Qwen/Qwen3-VL-Embedding-2B` |
| リランカー | `Qwen/Qwen3-VL-Reranker-2B` |
| train サンプル数 | 500 件 |
| eval サンプル数 | 200 件 |
| `max_pixels` | 200704 |
| `per_device_batch_size` | 2 |
| エポック数 | 1 |
| 損失関数 | MNRL（埋め込み）/ BinaryCrossEntropyLoss（リランカー）|

---

## トラブルシューティング経緯

### 問題 1: PIL Image の `add_column` エラー

`num_negatives > 0` のとき、PIL Image を `Dataset.add_column()` で追加しようとすると
PyArrow が型推論に失敗してクラッシュした。

```
pyarrow.lib.ArrowInvalid: Could not convert <PIL.PngImagePlugin.PngImageFile ...>
```

**修正**: `add_column` を廃止し、`train_reranker.py` と同様に `Dataset.from_dict` +
`HFImage()` Feature を明示して再構築する方式に変更した。

### 問題 2: `num_negatives=3` での VRAM 退避（22360 MiB）

`num_negatives=3` で各学習バッチに anchor + positive + negative × 3 = 5 テンソルが乗り、
VRAM が物理 16380 MiB を 5980 MiB 超過して共有メモリに退避。学習が 1 ステップ約 160 秒に
なり（通常は 10 秒程度）、`distill@ft_continue` で CUBLAS エラーが発生して失敗した。

```
CUBLAS_STATUS_EXECUTION_FAILED when calling cublasGemmEx(...)
```

`num_negatives=1` に下げて再実行したが、VRAM 退避は 21706 MiB と依然発生した。
`num_negatives` の有無（0 vs 1）よりも他の要因（後述）が VRAM を圧迫している。

---

## 結果

### 学習曲線

| ステップ | train loss |
|---|---|
| 10（epoch≈0.32） | 2.117 |
| 20（epoch≈0.64） | 1.015 |
| 30（epoch≈0.96） | 0.868 |

以前（accum=1）の学習曲線は step ごとに大きく振動していたが、今回は**単調降下**となり、
曲線の振動は改善された。ただしこれは有効ステップ数が 250 → 32 に激減したことによる
見かけ上の平滑化も含まれている可能性がある。

### 埋め込み単体評価

| モデル | MRR@10 | NDCG@10 | Acc@1 | Recall@10 |
|---|---|---|---|---|
| base | 0.261 | 0.110 | 0.143 | 0.038 |
| **finetuned（Issue #30）** | **0.655** | **0.597** | 0.571 | 0.194 |
| finetuned（Issue #33・今回） | 0.324 | 0.237 | 0.143 | 0.082 |

FT 後の精度が MRR@10 で 0.655 → 0.324（**−0.331**）と大幅に低下した。

### 2 段検索（rerank, 6 パターン）

| 構成 | MRR | NDCG@10 | 前回比 |
|---|---|---|---|
| embed=ft + rerank=ft | 0.441 | 0.252 | 前回 0.750（−0.309）|
| embed=ft + rerank=base | 0.310 | 0.218 | 前回 0.786（−0.476）|
| embed=ft + rerank=none | 0.389 | 0.235 | 前回 0.672（−0.283）|
| embed=base + rerank=ft | 0.411 | 0.141 | — |
| embed=base + rerank=none | 0.275 | 0.118 | — |
| embed=base + rerank=base | 0.244 | 0.116 | — |

### 知識蒸留

| variant | MRR@10 | NDCG@10 | 前回（Issue #30）MRR | 差分 |
|---|---|---|---|---|
| distill_oracle_ft | 0.347 | 0.197 | 0.806 | **−0.459** |
| distill_oracle_base | 0.405 | 0.183 | 0.386 | +0.019 |
| distill_ft_continue | 0.155 | 0.075 | 0.386 | −0.231 |
| distill_self | 0.216 | 0.152 | 0.310 | −0.094 |

`distill_oracle_base`（MRR=0.405）のみ前回比で微増した。FT 済みモデルを使う variant
（`oracle_ft`, `ft_continue`）は今回のFT品質低下をそのまま引き継いで悪化している。

---

## 原因分析

### 主因: 有効学習ステップ数の激減

`gradient_accumulation_steps: 1 → 8` により、**1 エポックの有効更新回数が 250 → 32 ステップ**
に激減した。1 エポック内で十分に収束しきれず、学習不足のまま終了した。

学習損失は 2.117 → 0.868 と降下しているが、32 ステップでは元の 250 ステップに比べて
最適化が不十分である。gradient accumulation はミニバッチの勾配を平均化して安定させる手法
だが、同一データを 1 回しか通らない場合、有効ステップ数の減少によるデメリットが大きい。

### 副因: VRAM 退避による学習時間の大幅増加

`num_negatives=1` でも VRAM 退避（21706 MiB）が発生し、1 ステップ約 100 秒に。
VRAM 退避は PCIe 越しのアクセスになるため速度が大幅に低下する（通常 10 秒/ステップ → 100 秒/ステップ）。
数値の正確性には影響しない。

VRAM 退避の原因は `num_negatives >= 1` によるバッチ内テンソル数増加（4 → 6 テンソル）で、
`num_negatives` の個数（1 vs 3）よりも「有無」が支配的な要因だった。

---

## 今後の対策

| 課題 | 対策案 |
|---|---|
| 有効ステップ数激減 | `gradient_accumulation_steps: 8 → 2〜4` に下げる（有効ステップ 62〜125 に増やす）|
| VRAM 退避（num_negatives=1） | `max_pixels: 200704 → 100352` に削減してバッチメモリを半減 |
| distill への grad_accum 伝播 | `dvc.yaml` の distill ステージの `--grad-accum` を `train.gradient_accumulation_steps` から切り離す |
| 学習曲線の可視化改善 | `logging_steps` を 5 前後に下げて、有効ステップ 32 でも複数点取れるようにする |

---

## 実行時間まとめ

| ステージ | 所要時間 | 前回比 | 備考 |
|---|---|---|---|
| train | 7225 秒（2.0 時間） | +5× | VRAM 退避（21706 MiB）が主因 |
| train_reranker | 1725 秒 | ほぼ同等 | 軽微な退避（18210 MiB）|
| distill@self | 10405 秒（2.9 時間） | ほぼ同等 | 以前からの既存問題 |
| distill@ft_continue | 完走 | — | 前回は CUBLAS エラー（num_negatives=3 時）|
| distill@oracle_base/ft | 各約 100 分 | ほぼ同等 | — |
