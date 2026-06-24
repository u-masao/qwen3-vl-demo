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

- ログ: `tmp/repro_seed1.log`（フル実行・OOM まで） / `tmp/repro_seed1_resume.log`（oracle 系再開） /
  `tmp/repro_rerankfix.log`（リランカー best 選択導入の再実行）。
- `dvc.lock` は seed 混在状態（oracle 系=seed1 / self・ft_continue=seed42 の旧記録）。
- メトリクス JSON: `outputs/metrics_base.json` / `metrics_finetuned.json` /
  `metrics_distill_oracle_{base,ft}.json` / `rerank_metrics.json`。

---

## 追記（2026-06-23）: リランカー学習にベストモデル選択を導入（対策①）

### 背景

考察 §3 のとおり、リランカーが検索精度を悪化させた引き金は **train_reranker が「最終
チェックポイント」をそのまま保存していた**こと（`save_steps`=10000 > 総 step、`eval_strategy`・
`load_best_model_at_end` なし）。極小バッチで乱高下するロスの“末尾のハズレ点”に着地していた。

### 変更

- `train_reranker.py`: 学習ペアの 10% を held-out 化し、`eval_strategy="steps"` /
  `load_best_model_at_end=True` / `metric_for_best_model="eval_loss"` を導入。`eval`・`save` を
  `eval_steps`(=50) に揃える。`per_device_eval_batch_size` は学習と同じ 2 に固定（OOM 回避）。
- `dvc.yaml`: `train_reranker` の cmd を `--save-steps` → `--eval-steps` に置換。

> ※ 学習中の選択指標に MRR/NDCG を使う案は見送り。eval が 7 persona で不安定なうえ、学習中に
> reranker 採点を繰り返すと distill で起きた採点 OOM を再誘発しうるため。held-out BCE loss で
> 「最終点ガチャ」を直接潰し、最終的な検索 MRR/NDCG は rerank ステージで測る方針とした。

### 結果（seed=1, FT 埋め込みに対するリランク）

| 構成 | 修正前（last ckpt） | 修正後（best=ckpt-50） |
|------|--------------------|----------------------|
| ft + none | MRR 0.830 / NDCG@10 0.549 / NDCG@1 0.660 | （同左・リランクなし） |
| ft + base | MRR 0.761 / 0.538 / 0.595 | （同左・base reranker） |
| ft + ft | MRR 0.655 / 0.510 / 0.385 | **MRR 0.665 / 0.531 / 0.465** |

- FT リランカーは小幅改善（NDCG@1 0.385→0.465、NDCG@10 0.510→0.531、MRR 0.655→0.665）。
- ただし **依然 ft+none(0.830) を下回り、base reranker(0.761) にも届かない**＝リランクは依然マイナス。
- 選ばれたのは **`checkpoint-50`（900 step 中の最初の評価点、eval_loss=0.510）**。
  以降 eval_loss は下がらず＝**リランカーは ~50 step で過学習**している。

### 解釈と次の一手

対策①は「最終点ガチャ」を解消する正しい方法論であり残す価値があるが、seed=1 の
「リランクが精度を悪化させる」問題自体は解けない。より深い原因は
(a) リランカーが即座に過学習（best=step50）、(b) 埋め込みが既に強く伸び代が無い、
(c) eval が 7 persona で高分散、の合わせ技。

- **過学習対策**: さらに短い学習 / 低 LR / weight decay / early-stopping 強化。
- **eval 解像度**: persona を増やしてリランカー可否を信頼性高く判定。
- **設計再考**: 強い bi-encoder（特に oracle_ft 蒸留 NDCG 0.622）前提なら、2 段リランク自体の
  費用対効果を見直す。

---

## 追記（2026-06-24）: eval の信頼化（per-persona マクロ集計＋不確かさの明示）

### 背景・方針

3 段構成（retriever→reranker→distill）を**信頼して比較できる評価基盤**にすることが目的。
本対策は **教師データ・タスク難易度を一切変えず**（プロジェクト方針：本格的な難易度化＝
ペルソナ/軸の手続き的生成は後続）、**集計と報告だけを正しくする**。学習系は再実行せず、
評価系のみ `dvc repro -s` で再計算（~7 分）。

問題は 2 つ: (1) eval が **7 persona ＝実効 7 クエリ**、(2) 200 行（画像単位）の**マイクロ平均**で
頻出ペルソナ（user_beta=43 枚）が指標を支配して偏る。

### 変更

- 新規 `metrics.py`: `ir_metrics` / `per_persona_metrics` / `macro_summary`（per-persona マクロ平均＋
  std/min/max＋ブートストラップ信頼区間）。`evaluate.py`・`rerank.py` 双方から再利用。
- `evaluate.py`: `build_ir_evaluator` のクエリを **ユニーク persona ごとに 1 つ**へ集約
  （IR 評価器の等重み平均＝per-persona マクロに）。`metrics_<label>_detail.json` に内訳・ばらつき出力。
- `rerank.py`: `rerank_metrics.json` を per-persona マクロ（後方互換キー）に。
  `rerank_metrics_detail.json` に per-persona 内訳・ばらつき・参考のマイクロ平均を出力。
- `dvc.yaml`: `metrics.py` を import 元ステージの dep に追加。

### 結果（per-persona マクロ。頻度バイアス除去で数値が変わる）

| モデル | macro MRR（旧 micro） | macro NDCG@10 | MRR の 95%CI |
|--------|---------------------:|--------------:|:-----------:|
| base | 0.284（0.354） | 0.119 | [0.12, 0.55] |
| finetuned | 0.786（0.823） | 0.505 | [0.57, 0.93] |
| distill_oracle_base | 0.239（—） | 0.136 | [0.08, 0.51] |
| **distill_oracle_ft** | **0.857** | **0.601** | [0.71, 1.00] |

リランク（FT 埋め込み, per-persona マクロ）: ft+none **0.786** → ft+base 0.719 → ft+ft 0.631
（**マクロでもリランカーは悪化**＝結論不変だが公平に測れた）。

### 主要な気づき

- **user_beta が易しすぎてマイクロを吊り上げていた**: base の per-persona MRR は user_beta=1.0 に対し
  他は 0.06〜0.25。マクロ化で偏りが消え、finetuned は 0.823→**0.786**、oracle_ft は **0.857** が正。
- **難しい persona が見えるように**: user_alpha / delta / eta が一貫して低い（retriever の弱点）。
- **不確かさを明示**: MRR の 95%CI は ±0.15〜0.25 と広い。**7 persona では検出力が低い**ことが
  数値で出た＝「絶対値や seed 間差を過信しない」根拠。

### 限界（意図的な切り分け）

集計の正しさ（頻度バイアス除去）と不確かさの明示までが本対策。**誤差バーの幅自体は persona 数に
律速**され、これ以上狭めるには後続の**手続き的生成によるペルソナ増（タスク難易度化）**が要る。
本対策はその土台（信頼できる物差し）を先に用意するもの。

### 補足

per-persona 内訳・ばらつき・CI は `outputs/*_detail.json`（`metrics_<label>_detail.json` /
`rerank_metrics_detail.json`）。`outputs/` は VCS 非追跡。図表（`figures.py` / `app.py`）への
per-persona・誤差バー表示は本対策では未着手（後続）。
