# seed=1 再現実験レポート（メインプロファイル切替の検証）

**実施日**: 2026-06-23
**ブランチ**: `main`
**関連 Issue**: [#25](https://github.com/u-masao/qwen3-vl-demo/issues/25)
**前回レポート**: [experiment_report_distillation_issue25.md](experiment_report_distillation_issue25.md)（seed=42）

---

## 概要

`common.seed` を 42 → **1** に変更し、フルパイプライン（`make repro`）を再実行した。
今後 seed=1 をメインプロファイルにする方針のため、その再現結果と seed=42 との差分・
異常の有無を確認することが目的。

**主な結果**:
- FT 埋め込みは明確に有効（base→finetuned で MRR@10 0.350→**0.823**）。seed=42 と同傾向。
- **`distill_oracle_ft` が NDCG@10=0.622・MAP@100=0.466 で全構成中最良**（finetuned の NDCG 0.545 を上回る）。
- **リランカーは MRR を悪化させる**（ft+none 0.830 → ft+ft 0.655）。既知の
  [reranker FT underperformance](reranker_ft_underperformance.md) を seed=1 でも再現。
- **OOM 再発**: `distill@self`（reranker teacher）の採点段階で num_negatives=7 でも OOM。
  Issue #25 の「7 は安全に通過」は再現せず。今回は oracle 系＋rerank のみ完走させた。

---

## 実験経緯

### seed 変更とフル再実行

`params.yaml` の `common.seed: 42 → 1`（commit `2d88d70`）を行い `nohup make repro` を実行。
seed 変更により画像生成キャッシュが無効化され、全 700 枚（train 500 / eval 200）を再生成した。

### distill@self での OOM 再発

`generate_data` → `eval_base` → `train` → `eval` → `train_reranker` まで完走した後、
`distill@self`（teacher=reranker）の **CrossEncoder teacher 採点（4000 ペア）に入った直後に OOM**。

- 失敗モード: VRAM が WSL2 共有メモリ（システム RAM）へスピル → ホスト RAM 枯渇 →
  Linux OOM killer が python3 を kill（`anon-rss ≈ 20GB`、make は exit 143/SIGTERM）。
- num_negatives=7（4000 ペア）でも安全マージンが無く、Issue #25 の安定動作は再現しなかった。

### 回避運用（oracle 系のみ完走）

完了済みステージは DVC キャッシュ済みのため、**reranker-teacher 系（`self`・`ft_continue`）を
スキップし、teacher モデル不要で VRAM 安全な oracle 系＋rerank だけを個別指定で再開**した:

```bash
dvc repro rerank \
  distill@oracle_base distill@oracle_ft \
  eval_distill@oracle_base eval_distill@oracle_ft
```

`generate_data` / `train` / `train_reranker` は "didn't change, skipping" で再利用され、
約 3.5 時間分の再計算を回避した。

---

## 実験設定

| パラメータ | seed=42（Issue #25） | seed=1（今回） |
|----------|---------------------|---------------|
| `common.seed` | 42 | **1** |
| `distill.num_negatives` | 7 | 7（変更なし） |
| `train.per_device_batch_size` | 2 | 2（変更なし） |
| `train.epochs` | 1 | 1（変更なし） |
| `preference.gamma` | 2.0 | 2.0（変更なし） |
| 実行した variant | self / ft_continue / oracle_base / oracle_ft | **oracle_base / oracle_ft のみ**（self・ft_continue は OOM でスキップ） |

---

## 結果

評価セット: eval 画像 200 枚。`evaluate.py` は各画像を 1 クエリ（`q0..q199`）として登録するが、
**クエリ文は persona 名**で、ユニークは **7 種**（user_alpha=8 〜 user_beta=43 枚と不均衡）。
各クエリの正解は同 persona の全画像（平均 ≈ 28 枚）の N:N relevance。
→ **実効的なクエリ多様性は 7 persona 分しかなく、指標は量子化・高ノイズ**である点に注意。

### 埋め込み検索（seed=1）

| モデル | MRR@10 | NDCG@10 | MAP@100 | Acc@1 | Acc@5 | Recall@10 |
|--------|--------|---------|---------|-------|-------|-----------|
| base | 0.350 | 0.135 | 0.099 | 0.215 | 0.660 | 0.044 |
| **finetuned** | **0.823** | 0.545 | 0.428 | 0.660 | 1.000 | 0.170 |
| distill_oracle_base | 0.305 | 0.187 | 0.115 | 0.215 | 0.365 | — |
| **distill_oracle_ft** | 0.795 | **0.622** | **0.466** | 0.590 | 1.000 | — |

### 2 段検索（rerank, 6 パターン）

| 構成 | MRR | NDCG@10 | NDCG@1 | Recall@10 |
|------|-----|---------|--------|-----------|
| embed=ft + rerank=none | **0.830** | 0.549 | 0.660 | 0.170 |
| embed=ft + rerank=base | 0.761 | 0.538 | 0.595 | 0.170 |
| embed=ft + rerank=ft | 0.655 | 0.510 | 0.385 | 0.170 |
| embed=base + rerank=none | 0.354 | 0.137 | 0.215 | 0.045 |
| embed=base + rerank=ft | 0.362 | 0.142 | 0.215 | 0.045 |
| embed=base + rerank=base | 0.269 | 0.126 | 0.000 | 0.045 |

### seed=42（Issue #25）との比較

| モデル | MRR@10 (seed42) | MRR@10 (seed1) | NDCG@10 (seed42) | NDCG@10 (seed1) |
|--------|----------------:|---------------:|-----------------:|----------------:|
| base | 0.273 | 0.350 | 0.133 | 0.135 |
| finetuned | 0.810 | 0.823 | 0.579 | 0.545 |
| distill_oracle_ft | 0.810 | 0.795 | 0.568 | **0.622** |
| distill_oracle_base | 0.323 | 0.305 | 0.198 | 0.187 |
| distill_self | 0.360 | —（未実行） | 0.154 | —（未実行） |
| distill_ft_continue | 0.697 | —（未実行） | 0.606 | —（未実行） |

---

## 考察

### 1. 主要傾向は seed=42 と一致

「FT が大きく効く」「oracle_ft が強い」「base 起点の oracle_base は base 同然」という構図は
両 seed で一致した。パイプラインの挙動は seed に対して安定しており、学習の崩壊・NaN・
退化は見られない。

### 2. oracle_ft 蒸留が並べ替え品質で最良

seed=1 では `distill_oracle_ft` が NDCG@10=0.622 / MAP@100=0.466 と全構成中最良で、
finetuned（NDCG 0.545）すら上回った（MRR はわずかに下 0.795）。FT 起点の oracle 蒸留は
bi-encoder の順位精度を底上げしており、最も有望な検索器。

### 3. リランカーが精度を悪化させる（要対応）

FT 埋め込みに rerank を足すと MRR が **0.830（none）→ 0.761（base）→ 0.655（ft）** と単調悪化。
ft reranker は NDCG@1 を 0.660→0.385 まで落とす。Recall@10 は 3 構成とも 0.170 で同一＝
**候補集合は同じで、リランカーが上位の並べ替えを誤っている**。
[reranker_ft_underperformance.md](reranker_ft_underperformance.md) の症状を seed=1 でも再現した。

### 4. 絶対値は seed=42 リファレンスをやや下回るが、主因は eval 規模

seed=1 のベストは ft+none の MRR 0.830・oracle_ft の NDCG 0.622 で、Issue #15 の参照値
（FT+reranker MRR 0.860 / NDCG@10 0.680）には届かない。ただし **eval は実効 7 persona** のため
絶対値の差は seed 揺れの寄与が大きい。一方「リランカーが悪化させる」方向は両 seed で一貫し、
こちらは実体のある課題。

### 5. num_negatives=7 でも OOM は再現する

reranker teacher 採点（CrossEncoder, Qwen3-VL-Reranker-2B）のピーク VRAM がスピルを誘発し、
ホスト RAM 枯渇で kill される。`predict` のバッチ（デフォルト 32）× `max_pixels=100352` が重い。
マイニング側の埋め込みモデルは採点前に `_free_model` で解放済みなので、2 モデル同居が主因ではない。

#### OOM 回避策の候補（優先順）

1. **採点バッチ縮小（本命）**: `distill.py` の `reranker.predict(chunk, batch_size=8, ...)`。ピーク VRAM を直接低減。num_negatives 維持＝条件不変。
2. `_SCORE_CHUNK`（現状 256）を 32〜64 に縮小し `empty_cache()` 頻度を上げる（①と併用）。
3. teacher 採点専用に `max_pixels` を下げる（`reranker.max_pixels` 直接変更は rerank に波及するため distill 専用パラメータ化）。
4. `num_negatives` を下げる（ピーク不変・累積/時間減・蒸留信号が変わる）。
5. **根治**: Windows NVIDIA ドライバ「CUDA - Sysmem Fallback Policy」を *Prefer No Sysmem Fallback* に（共有メモリ退避を止め、捕捉可能な CUDA OOM にする）＋ `.wslconfig` の memory/swap 拡張。
6. teacher を 8bit/4bit 量子化 or 小型化 / CPU 採点 / oracle で代替。

---

## 今後の方針

### 優先度 高

1. **seed=1 をメインプロファイルへ昇格**（本レポートの方針）。`params.yaml`（active）は反映済み、
   `default` プロファイルの source へも反映する。
2. **OOM 回避策 ①＋② を実装**し、self / ft_continue も seed=1 で揃えてフル比較を完成させる。

### 優先度 中

3. **リランカー underperformance の継続調査**（ハードネガティブ設計・クエリ設計）。現状は
   「リランクしない（ft+none）」が最良という結論を覆せていない。
4. **eval の解像度向上**: persona を 7 → より多くに増やし、指標の統計的信頼性を上げる
   （現状は実効 7 クエリで seed 揺れが大きい）。

### 優先度 低

5. oracle teacher の `temperature` 調整（soft label のシャープ化）。

---

## 補足: 実行メモ

- ログ: `tmp/repro_seed1.log`（フル実行・OOM まで） / `tmp/repro_seed1_resume.log`（oracle 系再開）。
- `dvc.lock` は seed 混在状態（oracle 系=seed1 / self・ft_continue=seed42 の旧記録）。
- メトリクス JSON: `outputs/metrics_base.json` / `metrics_finetuned.json` /
  `metrics_distill_oracle_{base,ft}.json` / `rerank_metrics.json`。
