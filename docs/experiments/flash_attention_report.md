# Flash Attention 有効化レポート

**実施日**: 2026-06-10  
**ブランチ**: `claude/nifty-feynman-ZW8gv`  
**対象 Issue**: [#2 bugfix: flash attention が効くようにして](https://github.com/u-masao/qwen3-vl-demo/issues/2)

---

## 1. 背景

`Qwen3-VL-Embedding-2B` のロード時に `attn_implementation="flash_attention_2"` を
指定することで推論を高速化できる。しかし、`flash-attn` パッケージが
`pyproject.toml` の `gpu` extras でコメントアウトされており、インストールしようとすると
ビルドエラーが発生していた。本レポートはその修正内容と性能検証結果をまとめる。

---

## 2. 問題: ビルドエラーの原因と修正

### 2.1 エラー内容

```
× Failed to build `flash-attn==2.8.3`
╰─▶ ModuleNotFoundError: No module named 'torch'
```

`flash-attn` の `setup.py` はビルド時に `torch` を import するが、uv はビルド用の
**隔離環境** を作るため、その環境に `torch` が含まれずビルドが失敗する。

### 2.2 修正内容

`pyproject.toml` に以下を追加した。

```toml
# flash-attn の setup.py がビルド時に torch を必要とするため明示する。
[tool.uv.extra-build-dependencies]
flash-attn = ["torch"]
```

これにより uv がビルド時に `torch` を隔離環境へ注入し、ビルドが成功する。

あわせて、`gpu` extras でコメントアウトされていた `flash-attn` を有効化し、
README にインストール手順を明記した。

```toml
[project.optional-dependencies]
gpu = [
    "flash-attn>=2.6.0; platform_system == 'Linux'",
]
```

インストール:

```bash
uv sync --extra gpu   # flash-attn 2.8.3 のビルド・インストール（初回は ~90 分）
```

### 2.3 フォールバック動作の改善

`models.py` のフォールバックログに、インストール案内を追加した。

```
WARNING: flash_attention_2 が使えません（…）。sdpa で再試行します
WARNING: flash-attn を有効にするには: uv sync --extra gpu   ← 追加
INFO:    attn_implementation: sdpa                           ← 実際に使われた実装
```

---

## 3. 性能ベンチマーク

### 3.1 計測環境

| 項目 | 内容 |
|---|---|
| GPU | NVIDIA RTX 4060 Ti 16 GB（Ada Lovelace） |
| PyTorch | 2.12.0+cu126 |
| flash-attn | 2.8.3 |
| モデル | Qwen/Qwen3-VL-Embedding-2B（bf16） |
| 入力 | 448×448 合成 RGB 画像 × 4 枚（~1024 パッチ/枚） |
| ウォームアップ | 2 回 |
| 計測 | 5 回平均 |

### 3.2 処理速度

| 実装 | 時間 (ms/batch) | eager 比 |
|---|---|---|
| `flash_attention_2` | 208 | **1.48× 高速** |
| `sdpa` | 206 | 1.47× 高速 |
| `eager` | 308 | 1.00×（基準） |

`flash_attention_2` と `sdpa` はほぼ同等。これは PyTorch 2.x の `sdpa`
（`scaled_dot_product_attention`）が Ada 世代 GPU で内部的に flash-attention
相当の最適化カーネルを使うためである。

### 3.3 VRAM 使用量

| 実装 | モデル本体 | 推論ピーク | 中間バッファ |
|---|---|---|---|
| `flash_attention_2` | 4,059 MB | 4,251 MB | 192 MB |
| `sdpa` | 4,059 MB | 4,251 MB | 192 MB |
| `eager` | 4,059 MB | 4,266 MB | 207 MB |

モデル本体サイズは実装によらず同一。推論中の中間バッファも 3 実装で 15 MB 以内の差しかない。
flash attention の「アテンション行列（O(n²)）を実体化しない」特性によるメモリ削減は、
系列長が数千トークンを超えた域で顕著になるため、今回の ~1024 パッチ規模では差が出なかった。

### 3.4 まとめ

| 観点 | 結論 |
|---|---|
| 速度 | flash_attention_2 ≈ sdpa、両者とも eager より **約 1.5× 高速** |
| VRAM | 3 実装の差は 15 MB 以内でほぼ同等 |
| 設定指針 | デフォルト `flash_attention_2` を維持。sdpa より有意に速くはないが、遅くもなく、longer-context シナリオへの準備として有効 |

---

## 4. 自動テスト

`tests/test_flash_attn_benchmark.py` に回帰テストを追加した。
CUDA GPU と flash-attn が利用可能な環境でのみ実行される。

```bash
uv sync --extra dev --extra gpu
uv run pytest tests/test_flash_attn_benchmark.py -v -s
```

**アサーション内容**: 448×448 画像 4 枚のバッチで、
`flash_attention_2` の encode 時間が `eager` の 1.20× 以上高速であること。

```
flash_attention_2 : 208 ms/batch
eager             : 308 ms/batch
speedup            : 1.48x  ✓ (閾値 1.20x)
```
