# Qwen3-VL マルチモーダル埋め込み ファインチューニング・デモ

**合成データだけで「画像検索の精度が上がる」体験**を、最小構成で一気通貫に再現するデモです。

1. 🎨 **データ生成** — 画像生成モデル [SD-Turbo](https://huggingface.co/stabilityai/sd-turbo) で、キャプション付き画像データセットを自動生成（キャプション＝そのまま検索クエリの正解になる）
2. 📐 **ベース評価** — [Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) のテキスト→画像検索精度（NDCG / Recall@k）を測定
3. 🔧 **ファインチューニング** — [Sentence Transformers](https://sbert.net) で埋め込みモデルを合成ペアに適応
4. 📈 **再評価** — 学習前後で検索精度を比較
5. 🥇 **リランク** — [Qwen3-VL-Reranker-2B](https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B) で上位候補を再ランクして仕上げ

```
caption (text)  ──SD-Turbo──▶  image
      │                          │
      └──────────  (text, image) ペア  ──────────┐
                                                 ▼
        Qwen3-VL-Embedding-2B  ──fine-tune──▶  改善した検索
                                                 ▼
                          Qwen3-VL-Reranker-2B で再ランク
```

> なぜ面白いか: **人手アノテーション不要**。画像生成のプロンプトがそのまま「正解ラベル付きのクエリ」になるので、
> 検索モデルの学習データがタダで無限に作れる、という発想のデモです。

---

## 必要環境

実際の学習・生成には **CUDA GPU が必須**です。本デモのデフォルト設定は
**NVIDIA RTX 4060 Ti 16GB（Ada 世代）** を想定しています。

- bf16（Ada はネイティブ対応）+ 勾配チェックポイント + 小バッチで 16GB に収まるよう調整
- `flash_attention_2`（未導入時は自動で `sdpa` にフォールバック）
- ディスク: モデルキャッシュ（SD-Turbo + Qwen3-VL 2B ×2）で十数 GB 程度

GPU が無い環境では、配線だけを確認できる [スモークテスト](#スモークテストgpu不要) を用意しています。

---

## セットアップ

[`uv`](https://docs.astral.sh/uv/) を使います。

```bash
uv sync                      # 依存をインストール
# GPU 機では CUDA 版 torch を入れてから（環境に合わせて）:
#   uv pip install torch --index-url https://download.pytorch.org/whl/cu124
#   uv sync --extra gpu      # flash-attn 等の GPU 専用 extra
```

> 注: `pyproject.toml` の `torch` は CPU でも import できるよう緩く指定しています。
> GPU 機では先に CUDA ビルドの torch を入れてください。

---

## 使い方

```bash
make all                     # フルパイプライン（GPU 推奨）
```

これは次を順に実行します（個別にも実行可）:

| ターゲット | 内容 |
|---|---|
| `make data`       | SD-Turbo でデータ生成 → `data/{train,eval}` に保存 |
| `make eval-base`  | ベースモデルの検索精度 → `outputs/metrics_base.json` |
| `make train`      | 埋め込みモデルを FT → `outputs/model/` に保存 |
| `make eval`       | FT 後の検索精度 → `outputs/metrics_finetuned.json` |
| `make rerank`     | 検索 top-k を Reranker で再ランク → `outputs/rerank_examples.json` |

完了後、`metrics_base.json` と `metrics_finetuned.json` を比べると NDCG / Recall の改善が確認できます。

### 設定の変更

`configs/default.yaml` を編集する（データ件数・バッチサイズ・モデル ID・画像トークン上限など）。
別ファイルを使う場合は各コマンドに `--config path/to.yaml` を渡せます。

```bash
uv run python -m qwen3vl_demo.train --config configs/default.yaml
```

---

## スモークテスト（GPU不要）

重いモデルをダウンロードせずに、パイプラインの配線（データ形式・Trainer・Evaluator・出力）が
通るかを CPU で確認します。

```bash
make smoke
```

スモークプロファイル（`configs/smoke.yaml`）では:

- 画像生成は **合成スタブ画像**（キャプションのハッシュで色を決める単色画像）に置換
- 埋め込みは小型の **`sentence-transformers/clip-ViT-B-32`**（CPU 可）に置換
- リランクは **スキップ**（小型のマルチモーダル cross-encoder が無いため）

⚠️ スモークは**配線確認専用**です。ここで出る数値に意味はありません。本番の精度は GPU で `make all` を実行してください。

---

## 画像生成プロンプトの作り方

学習データのキャプションは、[`src/qwen3vl_demo/prompts.py`](src/qwen3vl_demo/prompts.py) で
**手書きの単語リスト × 文テンプレート**を組み合わせて合成します（外部依存なし・seed で再現可能）。

- `SUBJECTS`（カテゴリ付き被写体: animal / vehicle / food / scene / object）
- `ADJECTIVES`（形容詞）／ `SETTINGS`（情景）／ `TEMPLATES`（文型）

例: `"a fluffy photo of a cat on a wooden table"`。この文がそのまま
① SD-Turbo へのプロンプト と ② 検索評価の正解クエリ の両方になります。
件数・seed は config で制御し、train と eval は別 seed で重複しないようにしています。

---

## 構成

```
src/qwen3vl_demo/
├── config.py         # YAML -> dataclass、--config / --profile
├── prompts.py        # テンプレート組み合わせでキャプション生成
├── generate_data.py  # SD-Turbo（or スタブ）で画像生成 → datasets 保存
├── models.py         # 埋め込みモデルのロード（attn フォールバック付き）
├── evaluate.py       # InformationRetrievalEvaluator で NDCG/Recall
├── train.py          # MultipleNegativesRankingLoss で FT
└── rerank.py         # 埋め込み検索 top-k → Reranker で再ランク
```

---

## 結果（例・差し替え用）

GPU で `make all` を実行後、ここに実測値を記入してください。

| モデル | NDCG@10 | Recall@1 | Recall@10 |
|---|---|---|---|
| ベース (Qwen3-VL-Embedding-2B) | _TBD_ | _TBD_ | _TBD_ |
| ファインチューニング後 | _TBD_ | _TBD_ | _TBD_ |
| ＋ Reranker-2B 再ランク | _TBD_ | _TBD_ | _TBD_ |

参考: Sentence Transformers 公式の Visual Document Retrieval 例では、同モデルの FT で
NDCG@10 が 0.888 → 0.947 に改善した報告があります。

---

## ライセンス

MIT（[LICENSE](LICENSE)）。使用する各モデル（Qwen3-VL, SD-Turbo, CLIP）のライセンスは各モデルカードを参照してください。
