# README 用の画像

README（`README.md` / `README.ja.md`）から参照しているサンプル画像を置く場所です。
これらの画像は**生成済みの成果物（`data/` と `outputs/model`）から作るため、リポジトリ
には含めず各自で生成**します（GPU 環境を想定）。

## ファイル一覧

| ファイル | 内容 | 生成方法 |
|---|---|---|
| `sample_grid.png` | 生成画像グリッド（カテゴリ横断のコンタクトシート） | `make figures`（モデル不要） |
| `retrieval_before_after.png` | 検索 Before/After（ベース埋め込み vs FT 埋め込み） | `make figures`（FT 済みモデルが必要） |
| `gradio_dataset.png` | Gradio「🖼️ データセット閲覧」タブのスクショ | 手動撮影（下記） |
| `gradio_metrics.png` | Gradio「📊 メトリクス比較」タブのスクショ | 手動撮影（下記） |

## 図の生成（`sample_grid.png` / `retrieval_before_after.png`）

`make all`（または最低でも `make data` と `make train`）を実行したあとに:

```bash
make figures
# もしくは: uv run qwen3vl-figures --profile default
```

`docs/images/` に PNG が書き出されます。FT 済みモデル（`outputs/model`）がまだ無い場合、
Before/After 図はスキップされ、グリッドだけが生成されます。

主なオプション（`uv run qwen3vl-figures --help`）:

- `--split {train,eval}` グリッドに使うスプリット（既定: `eval`）
- `--num-grid N` グリッドの枚数（既定: 12）
- `--num-queries N` Before/After に並べるクエリ数（既定: 3）
- `--top-k N` 1 クエリあたりの表示件数（既定: 5）

## Gradio スクリーンショット（`gradio_dataset.png` / `gradio_metrics.png`）

```bash
uv run python app.py        # → http://localhost:7860
```

ブラウザで開き、「🖼️ データセット閲覧」タブと「📊 メトリクス比較」タブを表示して
スクリーンショットを撮り、それぞれ `docs/images/gradio_dataset.png` /
`docs/images/gradio_metrics.png` として保存してください。
