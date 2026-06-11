# アーキテクチャ

このドキュメントは、デモ全体の構造・モジュール間の依存関係・データの流れを説明します。
個々の処理の中身（なぜそうするのか）は [動作解説](how-it-works.md) を、設定値の意味は
[仕様](specification.md) を参照してください。

---

## 1. 全体像

「合成データだけで画像検索の精度を上げる」一連の流れを 5 ステージに分割しています。
各ステージは独立した Python モジュールで、CLI（`python -m qwen3vl_demo.<module>`）として
単体実行できます。これらを **Makefile**（手軽に実行）または **DVC**（依存追跡つき再現実行）で束ねます。

```
┌─────────────┐   captions    ┌──────────────┐   (text, image)   ┌──────────────┐
│  prompts.py │ ────────────▶ │ generate_data │ ────────────────▶ │   datasets   │
│ (キャプション)│ + FLUX.2-klein│   .py         │   train / eval    │  (ディスク)   │
└─────────────┘               └──────────────┘                   └──────┬───────┘
                                                                         │
                          ┌──────────────────────────────────────────────┤
                          ▼                                              ▼
                  ┌──────────────┐  metrics_base.json        ┌──────────────┐
                  │  evaluate.py │ ◀── ベース評価             │   train.py   │
                  │ (IR 評価器)   │                            │  (MNRL でFT)  │
                  └──────┬───────┘                            └──────┬───────┘
                         │ metrics_finetuned.json (FT 後評価)        │ outputs/model
                         ▼                                          ▼
                  ┌────────────────────────────────────────────────────────┐
                  │                       rerank.py                         │
                  │  FT 済み埋め込みで top-k 検索 → Reranker-2B で並べ替え      │
                  └────────────────────────────────────────────────────────┘
                                          │ rerank_metrics.json / rerank_examples.json
                                          ▼
                                  ┌──────────────┐
                                  │    app.py    │  Gradio で結果を可視化
                                  └──────────────┘
```

---

## 2. モジュール構成と責務

| モジュール | 責務 | 主な入力 | 主な出力 |
|---|---|---|---|
| `config.py` | YAML 設定を dataclass に読み込み、共通 CLI 引数（`--config`/`--profile`）を提供 | `configs/*.yaml` | `Config` オブジェクト |
| `prompts.py` | テンプレート組み合わせでキャプションを生成（決定的） | 件数・seed | `list[Sample]` |
| `generate_data.py` | キャプションから画像を生成し datasets 化して保存 | `Config` | `data*/train`, `data*/eval` |
| `models.py` | 埋め込みモデルのロード（attention フォールバック付き） | `Config`, model_id | `SentenceTransformer` |
| `evaluate.py` | テキスト→画像検索の精度を IR 評価器で測定 | `data*/eval`, モデル | `metrics_*.json` |
| `train.py` | MNRL で埋め込みモデルをファインチューニング | `data*/train`, `Config` | `outputs*/model` |
| `train_reranker.py` | 負例マイニング＋BCE でリランカーをファインチューニング | `data*/train`, `Config` | `<reranker.model_dir>` |
| `rerank.py` | 埋め込み検索 top-k を Reranker（FT 済み優先）で再ランク | `data*/eval`, モデル | `rerank_examples.json` |
| `app.py` | 成果物（メトリクス・データ・リランク結果）を Gradio で可視化 | `data*/`, `outputs*/` | Web UI |

### 依存関係（import グラフ）

```
config.py   ← すべてのモジュールが依存（設定・パス解決）
models.py   ← evaluate.py / train.py / rerank.py（モデルロードを共通化）
evaluate.py ← train.py（build_ir_evaluator を学習中の途中評価に再利用）
prompts.py  ← generate_data.py
```

ポイント:
- **`models.py` の共通化**: 学習と評価で「同じ手順」でモデルを構築するため、ロード処理を 1 箇所に集約。
- **`evaluate.build_ir_evaluator` の再利用**: 学習スクリプトが独自に評価器を組まず、評価モジュールの実装を共有することで、学習中の途中評価と最終評価の指標定義がズレないようにしています。

---

## 3. データの形（コントラクト）

各モジュールが受け渡す「データの形」を固定しておくことで、モジュールを差し替えやすくしています。

### データセット（`generate_data.py` → `evaluate.py` / `train.py`）

`datasets.Dataset` を `save_to_disk` で永続化。カラム:

| カラム | 型 | 意味 |
|---|---|---|
| `anchor` | `string` | キャプション（画像生成プロンプト）— 学習時は使用しない |
| `positive` | `Image` | レンダリングされた画像（＝検索ターゲット） |
| `category` | `string` | 被写体カテゴリ（animal/vehicle/food/scene/object） |
| `subject` | `string` | 被写体単語（`"cat"` 等）— カテゴリより細粒度 |
| `persona` | `string` | ペルソナ名（`"user_alpha"` 等）— FT と評価でのクエリに使用 |

> FT の学習では `persona` を anchor として使用（train.py が `persona` 列を `anchor` に置換）。
> `anchor` / `positive` という名前は Sentence Transformers の対照学習が期待する慣習に合わせています。

### メトリクス JSON（`evaluate.py` → `app.py`）

`InformationRetrievalEvaluator` が返す dict をそのまま保存。キーは
`synthetic-image-retrieval_cosine_<metric>`（例: `..._ndcg@10`）の形式。`app.py` は
この接頭辞を剥がして表示します。

### リランク事例 JSON（`rerank.py` → `app.py`）

各ペルソナ代表クエリについて `query` / `num_relevant` / `best_rank_before_rerank` / `best_rank_after_rerank` / `hits_in_topk_before` / `hits_in_topk_after` / `top_k` を記録した配列。マルチポジティブ設定に対応し、正解集合の中での最良順位と top-k 内ヒット数を記録する。

---

## 4. 実行の束ね方: Makefile と DVC

同じ CLI を 2 通りの方法で起動できます。

- **Makefile** … 手軽に叩く用。`make all` / `make smoke` / 各ステージ個別。
- **DVC**（`dvc.yaml`） … 依存追跡つきの再現実行用。各ステージの `deps`（ソース・入力データ）と
  `outs`/`metrics`（出力）を宣言してあり、`dvc repro` で **変更があったステージだけ** を再実行します。
  `foreach` で `default` / `smoke` の 2 プロファイルを同一定義から展開しています。

両者は同じ `python -m qwen3vl_demo.*` コマンドを呼ぶだけなので、挙動は一致します。

---

## 5. プロファイル: default と smoke

GPU の有無で実行内容を切り替えるために 2 つのプロファイル（= 2 つの YAML）を用意しています。

| 観点 | `default`（本番・GPU） | `smoke`（配線確認・CPU） |
|---|---|---|
| 画像生成 | FLUX.2-klein-4B で実生成 | スタブ画像（ハッシュ由来の単色） |
| 埋め込みモデル | Qwen3-VL-Embedding-2B | clip-ViT-B-32（小型・CPU 可） |
| リランカー | Qwen3-VL-Reranker-2B（FT＋推論） | なし（学習・推論ともスキップ） |
| 件数 | train 500 / eval 200 | train 8 / eval 4 |
| dtype / device | bf16 / cuda | float32 / cpu |

プロファイル分岐は基本的に **設定値の差**で表現し、コード分岐は最小限（スタブ画像の使用と
混合精度の有無くらい）に留めています。これにより「smoke で配線を確認 → default で本番」という
流れが、同じコードパスで保証されます。

プロファイルは YAML を増やすだけで追加できます。例として `flux`（`configs/flux.yaml`）は
画像生成を VRAM 節約版の FLUX.2-klein-4b-fp8 に差し替えたプリセットで、`--profile flux` で使えます。
