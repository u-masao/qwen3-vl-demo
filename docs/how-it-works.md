# 動作解説（データ生成・学習・評価）

各ステージが「何を・なぜ・どうやって」行うのかを掘り下げて説明します。
構造の俯瞰は [アーキテクチャ](architecture.md)、設定値の一覧は [仕様](specification.md) を参照してください。

このデモの根本アイデアは 1 つです:

> **画像を生成するための指示文（プロンプト）は、そのまま「正解ラベル付きの検索クエリ」になる。**

text→image の画像生成では「テキスト T から画像 I を作る」ので、(T, I) は意味的に対応した
ペアです。検索の学習では「クエリ T に対する正解文書 I」のペアがほしい。つまり **画像生成の
入出力をそっくり検索の学習データに転用できる**わけです。人手アノテーションなしに学習データを
いくらでも作れる、というのがこのデモの肝です。

---

## ステージ 1: データ生成（`generate_data.py`）

### やること
キャプションを生成 → それを SD-Turbo に渡して画像を 1 枚ずつレンダリング → 
`datasets.Dataset`（`anchor`=キャプション, `positive`=画像, `category`=カテゴリ）に詰めて
train / eval に分けて保存します。

### キャプションの作り方（`prompts.py`）
手書きの語彙（被写体 `SUBJECTS` / 形容詞 `ADJECTIVES` / 情景 `SETTINGS`）と
文テンプレート `TEMPLATES` を `random.Random(seed)` で組み合わせて 1 文を作ります。

```
TEMPLATES 例:  "a {adj} photo of a {subj} {setting}"
              ↓ adj=fluffy, subj=cat, setting=on a wooden table
生成キャプション: "a fluffy photo of a cat on a wooden table"
```

- **決定的**: 同じ seed なら毎回同じキャプション集合。データの再現性が保てます。
- **重複排除**: 生成済みの文は集合で弾き、一意なキャプションだけを返します。
- **train と eval は別 seed**（`seed` と `seed + 10000`）で、両スプリットのキャプションが
  重ならないようにしています（評価の妥当性のため）。
- **カテゴリを保持**: 各キャプションに被写体カテゴリを添えておき、評価で「同カテゴリも正解」と
  する緩い設定（`relevant_same_category`）に使えるようにしています。

### 画像のレンダリング
- `diffusers` の `AutoPipelineForText2Image` で SD-Turbo をロード。
- SD-Turbo は蒸留モデルなので **`num_inference_steps=1`、`guidance_scale=0.0`** という
  超高速設定で 1 枚あたり一瞬。大量のペアを現実的な時間で作れます。
- seed 固定の `torch.Generator` を使い、同じ設定なら同じ画像が出る（再現性）。

### スタブ画像（smoke モード）
`smoke` プロファイル、または `image_gen.model_id="stub"` のときは、拡散モデルを
ダウンロードせず、**キャプションのハッシュから色を決めた単色画像**を返します。
「キャプションと画像が 1 対 1 で対応する」という前提だけは満たすので、CPU だけで
パイプライン全体の配線を確認できます（ただし数値に意味はありません）。

### 出力
`<data_dir>/train` と `<data_dir>/eval`。`positive` を `datasets.Image` 型にしてあるので、
保存時に画像が適切にシリアライズされ、読み込み時に PIL 画像として自動デコードされます。

---

## ステージ 2: 学習（ファインチューニング）（`train.py`）

### 損失関数: MultipleNegativesRankingLoss（MNRL）
このデモの中心です。MNRL は **明示的な負例を用意せず**、(anchor, positive) ペアだけで
対照学習を行います。仕組み:

```
バッチ内に N 件の (caption_i, image_i) があるとき:
  ・caption_i の正例 = image_i           （対応する画像）
  ・caption_i の負例 = image_j (j ≠ i)    （同じバッチの他の画像すべて）
目標: sim(caption_i, image_i) を上げ、sim(caption_i, image_j) を下げる
```

- バッチ内の他サンプルを負例に流用する（in-batch negatives）ため、**バッチサイズが
  大きいほど 1 サンプルあたりの負例が増え、学習が効きやすくなります**（VRAM と要相談）。
- テキストと画像を同じ埋め込み空間に近づける＝**クロスモーダル検索**を直接最適化します。

### モデルのロード（`models.py`）
- `SentenceTransformer("Qwen/Qwen3-VL-Embedding-2B", ...)` をロード。
- GPU では `dtype`（既定 bf16）と `attn_implementation`（既定 flash_attention_2）を指定。
  flash-attn が無ければ自動で `sdpa` → モデル既定へフォールバックするので、環境差で落ちません。
- `max_pixels` を指定すると画像のパッチ数を抑え、VRAM を節約できます。

### 16GB GPU に収めるための工夫
- **bf16**（Ada ネイティブ）で活性値・勾配を半精度に。
- **勾配チェックポイント**（`gradient_checkpointing`）で活性値を保持せず再計算し、メモリを節約。
- **小バッチ**＋必要なら `gradient_accumulation_steps` で実効バッチを確保。
- OOM が出たら、`per_device_batch_size` を下げる → `max_pixels` を下げる → `image_size` を下げる、の順で調整。

### 学習中の途中評価
`evaluate.py` の `build_ir_evaluator` で作った評価器を `Trainer` に渡し、`eval_steps` ごとに
検索精度を測ります。学習の最終評価と**同じ指標定義**を使うので、推移と最終結果が地続きです。

### 出力
学習後のモデルを `outputs/model/`（`cfg.model_path`）に `save_pretrained` で保存します。

---

## ステージ 3: 評価（`evaluate.py`）

### 評価の構図
`InformationRetrievalEvaluator` を使い、**テキストでクエリして画像を検索**する設定で測ります。

```
queries       : 各 eval 行のキャプション（テキスト）  q0, q1, ...
corpus        : 全 eval 画像                         d0, d1, ...
relevant_docs : q_i の正解 = d_i （厳密 1 対 1）
                ※ relevant_same_category=true なら同カテゴリの画像も正解に追加
```

評価器はクエリ（テキスト）とコーパス（画像）をそれぞれ埋め込み、コサイン類似度で全件を
ランキングして、NDCG / Recall / MRR などを @k で算出します。

### ベース vs ファインチューニング後
- `--label base`（既定）でベースモデルを評価 → `metrics_base.json`
- `--finetuned` で `outputs/model/` の FT 済みモデルを評価 → `metrics_finetuned.json`

2 つの JSON を比べると、ファインチューニングによる NDCG / Recall の改善が確認できます
（参考: Sentence Transformers 公式の Visual Document Retrieval 例では NDCG@10 が
0.888 → 0.947 に改善した報告があります）。

### なぜ改善するのか
合成データのキャプションは特定の語彙・言い回し・被写体に偏っています。ファインチューニングで
埋め込み空間がこの分布に適応し、「このデータセット上での」テキストと画像の対応付けが鋭くなる
ため、検索精度が上がります（＝ドメイン適応）。

---

## ステージ 4: リランク（`rerank.py`）

評価とは別に、実運用に近い **retrieve-then-rerank（2 段階検索）** を体験できます。

1. **retrieve（粗く速く）**: FT 済み埋め込みで各クエリの上位 `top_k` 画像を取得。
   埋め込みは「クエリと文書を別々にベクトル化して内積」なので高速ですが粗い。
2. **rerank（精密に）**: その `top_k` 件だけを `CrossEncoder`（Qwen3-VL-Reranker-2B）で
   クエリと 1 件ずつペアにして精密スコアリングし、並べ替える。重いが高精度。

各クエリについて、正解画像が **リランク前後で何位だったか**を記録し
（`rerank_examples.json`）、順位が上がっていれば 2 段階構成の効果が見えます。
`reranker.model_id` が null（smoke）のときは自動でスキップします。

---

## ステージ 5: 可視化（`app.py`）

生成済みの成果物を Gradio で閲覧する読み取り専用ビューア（学習はしません）。

- **メトリクス比較**: `metrics_base.json` vs `metrics_finetuned.json` を棒グラフ＋差分表で。
- **データセット閲覧**: 生成画像とキャプション・カテゴリを 1 枚ずつブラウズ。
- **Reranking デモ**: `rerank_examples.json` のリランク前後の順位を表で比較。

```bash
uv run python app.py   # → http://localhost:7860
```

---

## まとめ: 一気通貫の流れ

```
prompts.py が文を作る
   → generate_data が画像を作り (text, image) ペアにする
      → evaluate がベース精度を測る（before）
         → train が MNRL でそのペアに適応させる
            → evaluate が FT 後精度を測る（after）→ before と比較
               → rerank が 2 段階検索で仕上げる
                  → app.py で全部を可視化
```

「画像生成の入出力 = 検索の学習データ」という 1 つの発想だけで、データ作成から
精度改善の検証までを自己完結させているのがこのデモの面白さです。
