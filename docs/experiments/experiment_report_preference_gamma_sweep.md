# 実験レポート: 潜在嗜好モデル（preference）と gamma スイープによるリランカー伸びしろの検証

**実施日**: 2026-06-14
**ブランチ**: `claude/upbeat-mccarthy-93siqw`
**関連 Issue**: #15

---

## 1. 背景

前々回（`docs/experiments/experiment_report_subject_query.md`）・前回（`docs/experiments/experiment_report_persona_query.md`）の
ペルソナ検索タスクは、「視覚的に無関係な subject をペルソナへ恣意的に割り当てる」設計だった。
これによりベースモデルの精度はランダム近くまで落とせたが、**リランカーの伸びしろが消える**という
別の問題が残った（`docs/experiments/reranker_ft_underperformance.md`）。

原因は relevance の構造にある。subject タスクの relevance は「persona⇔subject の二値マップ」で、
ある画像が persona に属するかどうかは**加法的（属性ごとに独立）に決まる**。内積で候補をスコアする
bi-encoder（埋め込み）はこの加法的な境界を 1 エポックでほぼ暗記でき（NDCG@10≈0.985 で飽和）、
クエリと候補を結合して見る cross-encoder（リランカー）が追加で学べる構造が残らない。実際、
subject タスクではリランカー FT が埋め込み単体に対して **−0.004〜+0.006** しか動かなかった。

そこで Issue #15 では、**人間の嗜好の構造**を写し取った潜在嗖好モデル（`data.task=preference`）を
導入した。狙いは「リランカーが本当に効く＝非加法的な構造」をタスクに埋め込み、2 段階検索の
価値を実証できる評価ベンチにすることである。

---

## 2. 目的

1. 潜在嗖好モデル（preference）の **gamma（非加法的交互作用の強度）** を 2.0 と 0.0 で振り、
   gamma がリランカーの伸びしろを生む中心ノブであることを定量検証する。
2. gamma>0 で **リランカー FT が埋め込み単体を上回る（reranker helps）** ことを示す。
3. gamma=0（加法的）で **伸びしろが消える（旧タスクの再現）** ことを確認する。
4. 上記をもって preference（gamma=2.0）を**既定タスクへ昇格**する判断材料とする。

---

## 3. 方法

### 3.1 潜在嗖好モデル（`src/qwen3vl_demo/preference.py`）

relevance の作り方を一手に引き受ける単一の真実（SSOT）。`generate_data` が各画像の潜在属性を
サンプルし、**その画像を最も好むペルソナ（argmax appeal）** をラベル（`persona` 列）に付ける。
これにより評価・学習・リランクの下流コードは「persona 列の一致＝関連」という従来の仕組みのまま、
一切変えずに新タスクへ切り替えられる（subject タスクと**並列**のバリアント）。

嗖好スコアは次式で定義される（`a` は候補の二値潜在属性、`2a-1` は中心化）:

```
appeal(persona, a) = θ_p·(2a-1)  +  γ·Σ coef·(a_i AND a_j)  +  λ·θ_global·(2a-1)  +  ε
                     └ 線形：潜在嗖好  └ 非加法：AND 交互作用   └ 人気バイアス       └ 個人ノイズ
```

| 構成要素 | 役割 |
|---|---|
| 潜在軸（7軸） | warmth / era / ornament / mood / saturation / material / setting（各二値） |
| アーキタイプ（6型） | 軸上の sparse な嗖好ベクトル（共有構造）。将来の未知ペルソナ few-shot 汎化の足場 |
| ペルソナ＝混合 | 各ペルソナは少数アーキタイプの convex 混合（あえて重なりを持たせ「似て非なる」候補を作る） |
| **非加法交互作用（γ）** | `coef·(a_i AND a_j)` を γ 倍で加算。負係数は「単体では好きだが両方そろうと嫌い」を表現 |
| 人気バイアス（λ） | 平均的嗖好に沿う候補は万人受け |
| 個人ノイズ（σ） | 決定的（同一属性→同一値）な揺らぎ＝「緩い一貫性」 |

**核心**: 非加法項 `a_i AND a_j` は bi-encoder が内積で表現しづらく、cross-encoder が得意とする。
したがって**リランカーの伸びしろはこの交互作用に由来する**。`gamma` がその強度ノブで、
`gamma=0` なら加法的＝伸びしろ≈0（旧タスクの再現）、`gamma>0` で伸びしろが出る。

クエリは opaque トークン（`"user_alpha"` 等）のままなので、ベースモデルはトークンと嗖好の
対応を知らず `base ≈ random` が保たれ、FT の効果を測れる。

### 3.2 gamma スイープの検証手順

`verify_issue15/run.sh` で `task=preference` の `dvc repro` を gamma=2.0 → gamma=0.0 の順で実行した。

**重要な設計知見**: 画像生成プロンプト（`s.text`）は **gamma 不変**で、persona ラベル
（argmax appeal）だけが gamma 依存。したがって gamma=2.0 で全画像をキャッシュすれば、
gamma=0.0 は FLUX 生成が全ヒットで省ける（13583s → 3979s）。生データは `verify_issue15/`
（VCS 非追跡）に各 gamma の `metrics_base/finetuned.json`・`rerank_metrics.json`・
`rerank_examples.json`・`preference_model.json` として退避してある。

### 3.3 評価設定

| 項目 | 設定 |
|---|---|
| 生成モデル | `black-forest-labs/FLUX.2-klein-4B`（bf16, steps=4, guidance=1.0, batch=1） |
| 埋め込みモデル | `Qwen/Qwen3-VL-Embedding-2B`（MNRL, 1 epoch, lr 2e-5, bs 2, GC 有効） |
| リランカー | `Qwen/Qwen3-VL-Reranker-2B`（HNM + BCE, num_negatives=3, 1 epoch, max_pixels=100352） |
| データ | train 500 / eval 200（7 ペルソナ） |
| preference knob | gamma ∈ {2.0, 0.0}, lam=0.3, sigma=0.1, sharpness=2.0（seed=42, 決定的） |
| 評価指標 | NDCG@{1,5,10} / MRR / Recall@k（マルチポジティブ） |
| 環境 | RTX 4060 Ti 16GB, WSL2, flash_attention_2 |

eval は 200 行それぞれをクエリ（クエリ文＝そのペルソナ名）に展開し、正解は同一ペルソナの全画像
（argmax-appeal ラベル）とするマルチポジティブ設定。ペルソナあたりの正解数は **10〜59 件
（平均 ≈29）** と不均一（嗖好に忠実なほど特定ペルソナに偏る）。

---

## 4. 結果

### 4.1 計算時間（検証実行）

| gamma | 所要時間 | spill peak | 備考 |
|---|---|---|---|
| 2.0 | 約 3h46m | 6190 MB | 初回。FLUX 生成込み（13583s） |
| 0.0 | 約 1h6m | 6170 MB | 生成は全キャッシュヒット（~2.7h 省略） |

> 既定プロファイル昇格後の `dvc repro` は、上記 gamma=2.0 の結果が DVC run-cache にあるため
> **全ステージがキャッシュヒットで即時復元**され、GPU 再計算は発生しない。

### 4.2 埋め込みモデル単体評価（base vs FT）

| メトリクス | gamma=2.0 base | gamma=2.0 FT | gamma=0.0 base | gamma=0.0 FT |
|---|---|---|---|---|
| Accuracy@1 | 0.19 | **0.53** | 0.42 | **0.845** |
| Accuracy@10 | 0.765 | **1.000** | 0.910 | **1.000** |
| NDCG@10 | 0.133 | **0.647** | 0.244 | **0.706** |
| MRR@10 | 0.273 | **0.703** | 0.526 | **0.923** |
| MAP@100 | 0.106 | **0.492** | 0.139 | **0.577** |
| Recall@10 | 0.035 | **0.208** | 0.060 | **0.190** |

ランダムベースライン（7 ペルソナなら Acc@1 ≈ 1/7 ≈ 0.143）に対し、**gamma=2.0 の base Acc@1=0.19 は
ほぼランダム**。gamma=0.0 の base Acc@1=0.42 は明確にランダム超で、加法的な嗖好だと候補が視覚的に
よくまとまり base 埋め込みでも部分的に当たることを示す。**交互作用を強めるほど base はランダムに近づく**。

### 4.3 2 段階検索 6 パターン（top_k=10）

#### gamma=2.0（交互作用あり）

| 埋め込み | リランカー | NDCG@1 | NDCG@5 | NDCG@10 | MRR | Recall@10 |
|---|---|---|---|---|---|---|
| base | none | 0.190 | 0.089 | 0.122 | 0.273 | 0.030 |
| base | base | 0.000 | 0.052 | 0.094 | 0.131 | 0.030 |
| base | ft | 0.000 | 0.108 | 0.106 | 0.167 | 0.030 |
| ft | none | 0.530 | 0.644 | 0.646 | 0.703 | 0.208 |
| ft | base | 0.760 | 0.639 | 0.654 | 0.820 | 0.208 |
| **ft** | **ft** | **0.790** | **0.696** | **0.680** | **0.860** | 0.208 |

**Δ（ft+ft − ft+none）= リランカー FT の伸びしろ**: **MRR +0.157 / NDCG@1 +0.26 / NDCG@5 +0.052 /
NDCG@10 +0.035**。**リランカーが効く（reranker helps）。**

#### gamma=0.0（加法的）

| 埋め込み | リランカー | NDCG@1 | NDCG@5 | NDCG@10 | MRR | Recall@10 |
|---|---|---|---|---|---|---|
| ft | none | 0.845 | 0.705 | 0.705 | **0.922** | 0.190 |
| ft | base | 0.730 | 0.761 | 0.707 | 0.856 | 0.190 |
| ft | ft | **0.885** | **0.799** | **0.734** | 0.917 | 0.190 |

**Δ（ft+ft − ft+none）**: **MRR −0.005 / NDCG@1 +0.04 / NDCG@5 +0.094 / NDCG@10 +0.029**。
MRR では伸びしろが消える（むしろ僅かにマイナス）。

### 4.4 gamma 別のリランカー伸びしろ比較

| 指標（ft+ft − ft+none） | gamma=2.0 | gamma=0.0 |
|---|---|---|
| **MRR** | **+0.157** | **−0.005** |
| NDCG@1 | +0.26 | +0.04 |
| NDCG@5 | +0.052 | +0.094 |
| NDCG@10 | +0.035 | +0.029 |

---

## 5. 考察

### 5.1 中心仮説の検証（正直版）

- **条件1（完走）**: ✓ 両 gamma でフルパイプライン完走。
- **条件2（γ>0 でリランカー伸びしろ）**: ✓ gamma=2.0 は MRR・NDCG@1 とも明確に正
  （MRR +0.157、NDCG@1 +0.26）。最も鮮明な指標は **MRR と NDCG@1**。
- **条件3（γ=0 で差が消える）**: **MRR では消滅（−0.005）するが NDCG@10 は +0.029 残る**＝部分的。
  preference は加法でも埋め込みが飽和しきらない（gamma=0 でも NDCG@10=0.705 で頭打ちでない）ため、
  並べ替え下位の改善余地がリランカーに残る。
- **条件4（base≈random）**: gamma=2.0 のみ成立（Acc@1 0.19、random≈0.143）。gamma=0 は 0.42 で非ランダム。

→ **中心仮説（γ が伸びしろを生む）は支持される。主張は MRR ベースで述べるのが妥当。**

### 5.2 subject タスクとの対比

| タスク | embed FT NDCG@10 | リランカー FT 伸びしろ |
|---|---|---|
| subject（旧） | 0.985（飽和） | −0.004〜+0.006（ほぼ無し） |
| preference gamma=2.0 | 0.647（余地あり） | **MRR +0.157 / NDCG@10 +0.035** |

subject では埋め込み FT が暗記で飽和しリランカーの仕事が無かった。preference（gamma=2.0）は
非加法構造により埋め込みが飽和せず、**リランカーが本来担うべき役割（2 段階検索の価値）を実証できる**。
これが preference を既定へ昇格する根拠である。

### 5.3 base リランカーの振る舞い

gamma=2.0 では base 埋め込み上で base/ft リランカーが NDCG@1 を 0.19→0.0 に**悪化**させる
（検索自体が壊滅的でリランクのしようがない）。一方 ft 埋め込み上では base リランカーですら
NDCG@1 を 0.53→0.76 へ改善する。リランカーは「まともな候補集合」を前提に効くことを示す。

---

## 6. 今後の課題

1. **ペルソナ分離（train/eval）**: 現状は同一ペルソナが train/eval 両方にあり、評価が実質
   メモリテスト。未知ペルソナを eval に回し、嗖好構造の汎化（few-shot）を測る。アーキタイプ
   共有構造はこの足場として設計済み。
2. **graded relevance の活用**: `relevance_score`（sigmoid）・`threshold`・`temperature` は実装済みだが
   argmax ラベルでは未使用。graded NDCG で「どれだけ好むか」の連続性まで評価する。
3. **gamma の中間点スイープ**: 0.0 と 2.0 の 2 点のみ。0.5 / 1.0 / 1.5 を足し、伸びしろの単調性を確認する。
4. **NDCG@10 残差（gamma=0）の説明**: 加法でも NDCG@10 が +0.029 残る要因（並べ替え下位の構造）を切り分ける。

---

## 付録 A: 主要成果物

| 成果物 | パス |
|---|---|
| 潜在嗖好モデル定義 | `src/qwen3vl_demo/preference.py` / `data/preference_model.json` |
| ベース／FT 埋め込み評価 | `outputs/metrics_base.json` / `outputs/metrics_finetuned.json` |
| 6 パターン 2 段階検索評価 | `outputs/rerank_metrics.json` / `outputs/rerank_examples.json` |
| 検証ドライバ・生データ | `verify_issue15/`（run.sh・gamma2/・gamma0/。VCS 非追跡） |
| パイプライン定義 | `dvc.yaml` / `dvc.lock`（既定 = preference gamma=2.0） |

## 付録 B: ペルソナ別 正解数（gamma=2.0, eval）

| ペルソナ | 正解数 | リランク前 best_rank → リランク後 |
|---|---|---|
| user_alpha | 10 | 1 → 1 |
| user_beta | 36 | 2 → 1 |
| user_gamma | 20 | 5 → 3 |
| user_delta | 38 | 3 → 1 |
| user_epsilon | 59 | 1 → 1 |
| user_zeta | 22 | 1 → 3 |
| user_eta | 15 | 1 → 1 |

リランク後に best_rank が改善するペルソナ（beta 2→1、gamma 5→3、delta 3→1）がある一方、
悪化するペルソナ（zeta 1→3）もある。集約指標（MRR +0.157）はネットで正。
