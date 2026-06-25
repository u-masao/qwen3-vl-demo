# README 用の画像

README（`README.md` / `README.ja.md`）やドキュメントから参照する画像を置く場所です。
ここにある画像は**すべて `make figures`（= `qwen3vl-figures`）で生成・再生成でき**、リポジトリに
コミット済みです。`data/` と `outputs/` の成果物から作り、前提データが無い図は警告を出して
スキップされます。

Gradio 画面（`uv run python app.py`）は対話的に確認するものなので、スクリーンショットはここには
置きません（ビューアの機能は README を参照）。

## 自動生成（`make figures` / `qwen3vl-figures`）

`make all`（最低でも `make data`、図によっては `make train` / `make rerank`）を実行した
あとに `make figures`（= `uv run qwen3vl-figures --profile default`）で `docs/images/` へ
書き出されます。`outputs/` は VCS 非追跡なので、学習曲線・混同行列など `outputs/` 由来の図は
**学習を回した環境でのみ**生成できます。

| ファイル | 内容 | 必要な前提 |
|---|---|---|
| `sample_grid.png` | 生成画像グリッド（カテゴリ横断のコンタクトシート） | `data/{split}`（モデル不要・CPU 可） |
| `retrieval_before_after.png` | 検索 Before/After（ベース vs FT 埋め込み・緑枠=正解） | `data/eval` ＋ FT 済み `outputs/model` |
| `metrics_base_vs_ft.png` | 埋め込み Base vs FT のメトリクス比較棒グラフ | `outputs/metrics_{base,finetuned}.json` |
| `rerank_metrics.png` | 2 段階検索 主要 4 パターン（埋め込み × リランカー）比較 | `outputs/rerank_metrics.json` |
| `preference_archetypes.png` | アーキタイプ × 軸の定義ヒートマップ | `data/preference_model.json`（CPU 可） |
| `persona_embeddings.png` | 全ペルソナ × 軸の嗜好 θ ヒートマップ | `data/preference_model.json`（CPU 可） |
| `preference_pipeline.png` | 嗜好空間 → 属性 → 語片 → プロンプト → argmax ラベル | `data/preference_model.json`（CPU 可） |
| `preference_interactions.png` | 非加法的交互作用の円環グラフ（緑=好む / 赤=嫌う） | `data/preference_model.json`（CPU 可） |
| `appeal_decomposition.png` | appeal の寄与分解（線形 / 交互作用 / 人気バイアス） | `data/preference_model.json`（CPU 可） |
| `dataset_persona_counts.png` | ペルソナ別データ件数（argmax ラベルの不均衡） | `data/{split}`（CPU 可） |
| `rerank_rank_changes.png` | rerank 前後の正解順位の変化（スロープ） | `outputs/rerank_examples.json` |
| `retrieval_confusion.png` | 検索の混同行列（クエリ × 取得ペルソナ・base / FT） | `data/eval` ＋ FT 済み `outputs/model` |
| `training_curve.png` | 埋め込み FT の学習曲線（loss ＋ eval NDCG@10） | `outputs/checkpoints/checkpoint-*/trainer_state.json` |
| `training_loss_overview.png` | 3 ステージ（埋め込み/リランカー/蒸留）の loss 俯瞰 | 各 `outputs/*_checkpoints/.../trainer_state.json` |

> 前提データが揃わない図は警告ログを出してスキップされます（他の図は生成されます）。
> 嗜好構造図（`preference_*` / `persona_embeddings` / `appeal_decomposition` /
> `dataset_persona_counts`）は `data/preference_model.json` だけで描けるので **CPU でも**作れます。

主なオプション（`uv run qwen3vl-figures --help`）:

- `--split {train,eval}` グリッド・データ統計に使うスプリット（既定: `eval`）
- `--num-grid N` グリッドの枚数（既定: 12）
- `--num-queries N` Before/After に並べるクエリ数（既定: 3）
- `--top-k N` Before/After・混同行列の上位件数（既定: 5）
- `--pipeline-persona NAME` パイプライン図・appeal 分解図の代表ペルソナ（既定: 先頭ペルソナ）

## Gradio ビューア

```bash
uv run python app.py        # → http://localhost:7860
```

メトリクス比較・データセット閲覧・ペルソナ閲覧・Reranking デモ・2 段階検索・学習曲線の 6 タブを
ブラウザで対話的に確認できます（スクリーンショットはコミットしません）。学習曲線は「📈 学習曲線」
タブでも見られ、自動生成の `training_curve.png` / `training_loss_overview.png` と同じ図です。
