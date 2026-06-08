# 仕様

このデモの「何を・どんな前提で・どう設定して動かすか」を定義します。処理の意図は
[動作解説](how-it-works.md)、構造は [アーキテクチャ](architecture.md) を参照してください。

---

## 1. 目的とスコープ

### 目的
画像生成モデルが作る (キャプション, 画像) ペアを学習データとして使い、
マルチモーダル埋め込みモデル（Qwen3-VL-Embedding-2B）の **テキスト→画像検索** 精度を
ファインチューニングで向上させられることを、一気通貫で体験できるデモを提供する。

### スコープに含むもの
- 合成データ生成（SD-Turbo）
- 埋め込みモデルのファインチューニング（Sentence Transformers / MNRL）
- 検索精度の評価（学習前後の比較）
- リランカー（Qwen3-VL-Reranker-2B）のファインチューニング（負例マイニング＋BCE）と 2 段階検索デモ
- 結果の可視化（Gradio）

### スコープに含まないもの
- 本番運用向けのベクトル DB / サービング
- 大規模データセットや分散学習

---

## 2. 使用モデル

| 役割 | モデル | ライセンス | 備考 |
|---|---|---|---|
| 画像生成 | [`stabilityai/sd-turbo`](https://huggingface.co/stabilityai/sd-turbo) | [Stability AI Community License](https://stability.ai/license) | 1〜4 ステップの蒸留モデル。guidance なし |
| 埋め込み（FT 対象） | [`Qwen/Qwen3-VL-Embedding-2B`](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) | Apache-2.0 | テキスト・画像を同一空間に埋め込む |
| リランカー（FT＋推論） | [`Qwen/Qwen3-VL-Reranker-2B`](https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B) | Apache-2.0 | cross-encoder。クエリ×文書を精密スコア |
| 埋め込み（smoke 代替） | `sentence-transformers/clip-ViT-B-32` | MIT | 小型・CPU 可。配線確認専用 |

> ⚠️ **ライセンスに注意**: リポジトリのコードは MIT ですが、各モデルには独自のライセンスがあり、
> 生成物にも条件が及ぶ場合があります。特に **SD-Turbo は Stability AI Community License** で、
> 年商 100 万ドルを超える組織での商用利用には別途ライセンスが必要です。詳細は各モデルカードと
> [README のライセンス節](../README.md#license) を参照してください。

---

## 3. 動作要件

### ハードウェア
- **本番（`default`）**: CUDA GPU が必須。基準は **NVIDIA RTX 4060 Ti 16GB（Ada 世代）**。
  - bf16（Ada ネイティブ）＋ 勾配チェックポイント＋小バッチで 16GB に収まるよう調整。
  - flash-attn が無くても自動で `sdpa` → モデル既定へフォールバック（`models.py`）。
- **配線確認（`smoke`）**: GPU 不要。CPU のみで数十秒〜数分。

### ソフトウェア
- Python 3.10 以上
- パッケージ管理は [`uv`](https://docs.astral.sh/uv/)
- 主要依存: `torch`, `torchvision`, `sentence-transformers[image] >= 5.4`, `transformers >= 4.57`,
  `datasets`, `diffusers`, `accelerate`, `gradio`, `matplotlib`（詳細は `pyproject.toml`）
- Linux では `pyproject.toml` の `[tool.uv.sources]` により torch/torchvision を
  CUDA 12.6 ビルド（`pytorch-cu126`）から取得。

### ディスク
- モデルキャッシュ（SD-Turbo ＋ Qwen3-VL 2B ×2）で十数 GB 程度。

---

## 4. 設定ファイル仕様（`configs/*.yaml`）

設定はすべて YAML に集約し、`config.py` の dataclass にマッピングされます。
パスはリポジトリルートからの相対で記述します。

| セクション.キー | 型 | default 値 | 意味 |
|---|---|---|---|
| `profile` | str | `default` | プロファイル名（`default`/`smoke`） |
| `seed` | int | `42` | 乱数シード（データ生成・学習の再現性） |
| `device` | str | `cuda` | `cuda` / `cpu` |
| `dtype` | str | `bfloat16` | `float32` / `float16` / `bfloat16` |
| `paths.data_dir` | str | `data` | データセット保存先 |
| `paths.output_dir` | str | `outputs` | メトリクス等の出力先 |
| `paths.model_dir` | str | `outputs/model` | FT 済みモデル保存先 |
| `data.num_train` | int | `200` | 学習ペア数 |
| `data.num_eval` | int | `50` | 評価ペア数 |
| `data.image_size` | int | `512` | 生成画像の一辺 px |
| `data.relevant_same_category` | bool | `false` | 同カテゴリ画像も正解とみなすか（緩い評価） |
| `image_gen.model_id` | str | `stabilityai/sd-turbo` | `stub` でスタブ画像 |
| `image_gen.num_inference_steps` | int | `1` | 拡散ステップ数 |
| `image_gen.guidance_scale` | float | `0.0` | Turbo 系は 0.0 |
| `image_gen.batch_size` | int | `8` | 生成バッチ |
| `embedding.model_id` | str | `Qwen/Qwen3-VL-Embedding-2B` | 埋め込みモデル |
| `embedding.attn_implementation` | str | `flash_attention_2` | 失敗時は自動フォールバック |
| `embedding.max_pixels` | int\|null | `null` | 画像トークン上限（VRAM 節約） |
| `embedding.query_prompt_name` | str\|null | `null` | クエリ用 instruction prompt 名 |
| `reranker.model_id` | str\|null | `Qwen/Qwen3-VL-Reranker-2B` | `null` でリランクをスキップ |
| `reranker.top_k` | int | `10` | リランク対象の上位件数 |
| `train.epochs` | int | `1` | エポック数 |
| `train.per_device_batch_size` | int | `4` | バッチサイズ（MNRL の負例数に直結） |
| `train.gradient_accumulation_steps` | int | `1` | 勾配累積 |
| `train.learning_rate` | float | `2.0e-5` | 学習率 |
| `train.warmup_ratio` | float | `0.1` | ウォームアップ比率 |
| `train.gradient_checkpointing` | bool | `true` | VRAM 節約 |
| `train.eval_steps` / `save_steps` / `logging_steps` | int | `50`/`50`/`10` | 評価・保存・ログ間隔 |

---

## 5. CLI 仕様

全エントリポイント共通で `--config PATH`（優先）と `--profile {default,smoke}` を受け付けます。

| コマンド | 追加引数 | 説明 |
|---|---|---|
| `python -m qwen3vl_demo.generate_data` | — | データセット生成 |
| `python -m qwen3vl_demo.evaluate` | `--model ID/PATH`, `--finetuned`, `--label STR` | 検索精度評価。`--finetuned` で FT 済みモデルを評価 |
| `python -m qwen3vl_demo.train` | — | 埋め込みモデルのファインチューニング |
| `python -m qwen3vl_demo.train_reranker` | — | リランカーのファインチューニング（reranker.model_id が null ならスキップ） |
| `python -m qwen3vl_demo.rerank` | `--num-queries N` | リランクデモ（表示するクエリ数） |
| `python app.py` | — | Gradio ビューア（:7860） |

`pyproject.toml` の `[project.scripts]` により、`qwen3vl-generate-data` 等の
コンソールスクリプトとしても起動できます。

---

## 6. 出力仕様（成果物）

| パス | 生成元 | 内容 |
|---|---|---|
| `<data_dir>/train`, `<data_dir>/eval` | generate_data | datasets（anchor/positive/category） |
| `<output_dir>/metrics_base.json` | evaluate (base) | ベースモデルのメトリクス |
| `<output_dir>/metrics_finetuned.json` | evaluate (--finetuned) | FT 後のメトリクス |
| `<output_dir>/model/` | train | FT 済み埋め込みモデル（SentenceTransformer） |
| `<output_dir>/checkpoints/` | train | 学習中チェックポイント（最新 1 個） |
| `<reranker.model_dir>/` | train_reranker | FT 済みリランカー（CrossEncoder） |
| `<output_dir>/rerank_metrics.json` | rerank | 4 パターン（埋め込み{base,ft}×リランカー{base,ft}）の検索指標 |
| `<output_dir>/rerank_examples.json` | rerank | リランク前後の順位事例（最良の組） |

> `data/`・`outputs/`（および smoke 版）は `.gitignore` 済み。再現は設定とコードから可能です。

---

## 7. 評価指標

`InformationRetrievalEvaluator` が算出する標準的な検索指標を使用します。

- **NDCG@10** … 順位を考慮した正規化累積利得（主要指標）
- **Recall@k** … 上位 k 件に正解が含まれる割合（k = 1, 3, 5, 10）
- **MRR@10** … 最初の正解の逆順位の平均
- **Accuracy@k / MAP@100** … 評価器が併せて出力

正解の定義は `data.relevant_same_category` で切り替わります（既定は厳密 1 対 1）。
