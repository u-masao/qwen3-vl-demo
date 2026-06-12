# Qwen3-VL-Embedding マルチモーダルドキュメント埋め込み 調査メモ

**調査日**: 2026-06-11  
**目的**: 既存のテキスト・複数画像・複数動画を「ひとつのドキュメント」として埋め込む際の API 形式と実装方針の把握

---

## 1. 背景

現在の実装は **「テキストクエリ → 単一生成画像」** の検索パイプライン（SentenceTransformers経由）。

新しい要件：
- 入力側（ドキュメント）：テキスト + 複数画像 + 複数動画 → 1つの埋め込みベクトル
- クエリ側：テキストのみ（現状維持）
- 合成データ生成は不要（既存の実在ドキュメントを使用）

---

## 2. Qwen3-VL-Embedding の入力形式（3アプローチ比較）

### 2-A. SentenceTransformers（現在のプロジェクトが使用中）

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Qwen/Qwen3-VL-Embedding-8B")

# テキストのみ
model.encode("A woman playing with her dog.")

# 単一画像（URL or PIL Image）
model.encode("https://example.com/image.jpeg")

# テキスト + 単一画像
model.encode({"text": "説明文", "image": "https://..."})

# 複数画像（リスト渡し）
model.encode({"text": "説明文", "image": [img1, img2, img3]})

# バッチ処理
embeddings = model.encode([item1, item2, ...], convert_to_tensor=True)
```

**制限**：`fps` / `max_frames` / `total_pixels` パラメータは ST ラッパー経由では制御不可の可能性がある。動画サポートは未確認。

---

### 2-B. Raw API（`Qwen3VLEmbedder`）— 最も柔軟

```python
from models.qwen3_vl_embedding import Qwen3VLEmbedder

embedder = Qwen3VLEmbedder(
    "Qwen/Qwen3-VL-Embedding-8B",
    max_pixels=1800 * 32 * 32,       # 1画像あたりのピクセル上限
    total_pixels=10 * 768 * 32 * 32, # 動画全フレームの合計ピクセル上限
    fps=1,                            # 動画のサンプリングレート
    max_frames=64,                    # 動画の最大フレーム数
)

# テキスト + 複数画像 + 動画（ファイルパス）
embeddings = embedder.process([
    {
        "text": "説明文",
        "image": [
            pil_image1,
            "path/to/image2.jpg",
            "https://example.com/image3.jpeg",
        ],
        "video": "path/to/video.mp4",  # 自動で "file://" プレフィクス付与
        "instruction": "カスタム指示（省略可）",
    },
    {
        # 動画をフレームリストで渡す場合
        "video": [frame1_pil, frame2_pil, frame3_pil],
    },
    {
        # 複数動画
        "video": ["video1.mp4", "video2.mp4"],
    },
])
# → shape: (N, 4096)  ※8B モデルの場合
```

**`format_model_input()` のシグネチャ**:

```python
def format_model_input(
    text: Optional[Union[List[str], str]] = None,
    image: Optional[Union[List[Union[str, Image.Image]], str, Image.Image]] = None,
    video: Optional[Union[List[Union[str, List[Union[str, Image.Image]]]], str, List[Union[str, Image.Image]]]] = None,
    instruction: Optional[str] = None,
    fps: Optional[float] = None,
    max_frames: Optional[int] = None,
) -> List[Dict]:
```

---

### 2-C. vLLM（高スループット推論特化）

```python
from vllm import LLM

llm = LLM(
    model="Qwen/Qwen3-VL-Embedding-2B",
    runner="pooling",
    dtype="bfloat16",
    trust_remote_code=True,
)

# 入力形式（vLLM 固有）
vllm_inputs = [
    {
        "prompt": "<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...",
        "multi_modal_data": {"image": pil_image},  # 現状は単画像のみ
    }
]

outputs = llm.embed(vllm_inputs)
embeddings = [o.outputs.embedding for o in outputs]
```

**注意**：公式サンプルの `format_input_to_conversation()` は単画像のみ対応。複数画像・動画サポートには改修が必要。

---

## 3. 主要パラメータ一覧

| パラメータ | デフォルト値 | 計算式 | 意味 |
|---|---|---|---|
| `max_pixels` | 1,843,200 | `1800 × 32²` | 1画像あたりのピクセル上限 |
| `min_pixels` | 4,096 | `4 × 16² × 4` | 1画像あたりの最小ピクセル数 |
| `total_pixels` | 7,864,320 | `10 × 768 × 32²` | 動画全フレームの合計ピクセル上限 |
| `fps` | 1 | — | 動画フレームサンプリングレート |
| `max_frames` | 64 | — | 動画の最大フレーム数 |
| `max_length` | 8,192 | — | トークン長の上限 |
| `embedding_dim` | 4,096 (8B) / 2,048 (2B) | — | 出力埋め込みの次元数 |

### 複数画像時の VRAM 見積もり

現在のプロジェクト設定：`max_pixels = 200,704`（`params_default.yaml`）

画像N枚の場合のトークン数概算：
- 200,704 pixels ≈ 448×448 → 約 784 トークン/枚
- N=5枚：約 3,920 トークン（`max_length=8192` の約48%）
- 動画 64フレーム × `FRAME_MAX_PIXELS = 768×32² = 786,432` 相当 → `total_pixels` で制御

**RTX 4060 Ti 16GB** では `max_pixels` を下げる（例：`100,352 = 56 × 56 × 32`）か、バッチサイズを小さくすることで対処可能。

---

## 4. アプローチ別サポート状況

| 機能 | SentenceTransformers | Raw API | vLLM |
|---|---|---|---|
| テキスト | ✅ | ✅ | ✅ |
| 単一画像（PIL） | ✅ | ✅ | ✅ |
| 単一画像（URL/パス） | ✅ | ✅ | ✅ |
| **複数画像** | ✅（リスト渡し） | ✅ | 要改修 |
| **動画（ファイルパス）** | 未確認 | ✅ | 要改修 |
| **動画（フレームリスト）** | 未確認 | ✅ | 要改修 |
| fps / max_frames 制御 | 不可能（可能性高） | ✅ | vLLM設定で別途 |
| ファインチューニング | ST Trainer 対応 | カスタム実装必要 | 不向き |
| バッチスループット | 中 | 低 | 高 |

---

## 5. 現在のプロジェクトへの影響範囲

クエリ：テキストのみ / ドキュメント：テキスト + 複数画像 + 複数動画 / ファインチューニング継続

| ファイル | 影響度 | 主な変更内容 |
|---|---|---|
| `models.py` | **極小** | encode() の入力 dict のキーを変えるだけ（ST継続の場合）または Qwen3VLEmbedder ラッパーに差し替え（raw API）|
| `generate_data.py` | **大（全面改修）** | 合成画像生成 → 実在ドキュメントのロード形式に転換。既存のロジックは基本的に不要。 |
| `evaluate.py` | **小〜中** | corpus を `[{"text":..., "image":[...], "video":...}]` の dict リストに変更するだけ |
| `train.py` | **中** | 学習データの dict 形式変更。ST継続なら DataCollator はライブラリ側が処理。raw API 切り替えならカスタム実装必要。 |
| `rerank.py` | **中** | Qwen3-VL-Reranker の複数画像・動画対応を別途確認要 |
| `prompts.py` | **実質不要** | 合成キャプション生成のためのファイル。既存ドキュメント使用なら削除または放置でよい。 |
| `params_default.yaml` | **小** | `max_pixels` の調整（複数画像でトークン急増を防ぐ） |

---

## 6. 推奨アプローチ

### 推論のみ（ファインチューニングなし）

SentenceTransformers のまま使用。`model.encode({"image": [img1, img2], "text": "..."})` で対応。

### ファインチューニングも継続する場合（今回の要件）

**Raw API（`Qwen3VLEmbedder`）に切り替えることを推奨**。

理由：
1. `fps` / `max_frames` / `total_pixels` を明示的に制御できる（VRAM 管理上重要）
2. 動画サポートが明示的に実装されている
3. 複数画像・動画の collation が `_preprocess_inputs()` 内で処理済み
4. `Qwen3VLForEmbedding` クラスはファインチューニング可能な形で設計されている

切り替えコスト：`models.py` に `Qwen3VLEmbedder` ラッパーを導入（〜50行）し、`evaluate.py` / `train.py` の encode 呼び出し箇所を `embedder.process([{...}])` 形式に変更する。

### vLLM は使わない

現在の規模（RTX 4060 Ti 16GB、研究用）では vLLM のセットアップコストに見合わない。本番スケールで高スループットが必要になった時点で検討する。

---

## 7. 参考資料

- `Qwen3VLEmbedder` ソース: [QwenLM/Qwen3-VL-Embedding](https://github.com/QwenLM/Qwen3-VL-Embedding/blob/main/src/models/qwen3_vl_embedding.py)
- vLLM ノートブック例: `QwenLM/Qwen3-VL-Embedding` リポジトリ内 `cookbooks/`
- モデルカード: [Qwen/Qwen3-VL-Embedding-8B on HuggingFace](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B)
- 論文: arXiv:2601.04720
