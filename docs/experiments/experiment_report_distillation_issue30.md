# 蒸留 OOM 修正レポート（Issue #30）

**実施日**: 2026-06-25  
**ブランチ**: `fix/issue-30-distill-oom-batch-size`  
**関連 Issue**: [#30](https://github.com/u-masao/qwen3-vl-demo/issues/30)  
**前回レポート**: [experiment_report_distillation_issue25.md](experiment_report_distillation_issue25.md)

---

## 概要

Issue #25（num_negatives=7）で実現した蒸留パイプラインが、`distill@self`（teacher=reranker）で
VRAM OOM を再発させていた。本 Issue では OOM の根本原因を特定・修正し、全 4 variant
（`self` / `ft_continue` / `oracle_base` / `oracle_ft`）が正常完走できる状態に戻した。

また num_negatives=7 が CPU RAM OOM の原因であることも判明し、3 に戻した。

**主な結論**:

- `distill_oracle_ft` は `num_negatives=3` でも **MRR@10=0.806 / NDCG@10=0.634** を達成し、
  FT 単体（0.655/0.597）・最強リランクパターン（ft+rerank=base: 0.786）を上回る
- `distill_ft_continue`（reranker teacher）は `num_negatives=7` のときより大幅に低下（0.697→0.386）。
  reranker teacher は負例数への依存が高く、3 ではマージン情報が薄い
- `_free_model` の VRAM 解放バグ（`del model` だけでは解放されない）を修正

---

## OOM 原因の特定過程

### フェーズ 1: 最初の OOM（VRAM）

`distill@self` が CrossEncoder スコアリング中の chunk 125/125 で無言で死亡。

デバッグログを段階的に追加した結果：

| ログ追加位置 | v1 での出力 | v2 での出力 |
|---|---|---|
| mine 後 VRAM flush | allocated=4.3GB | allocated=0.0GB |
| CrossEncoder ロード完了 | allocated=8.5GB | allocated=4.3GB |
| チャンク 125/125 採点開始 | 出力あり（その後停止） | 出力あり（その後停止）|
| チャンク 125/125 採点完了 | **出力なし** | **出力なし** |

**原因**: `_free_model` が `del model` のみで GPU VRAM を解放していなかった。
PyTorch アロケータはローカル変数を削除しても VRAM ブロックを保持し続けるため、
`mine_hard_negatives` 後に 4.3 GB が残留したまま CrossEncoder（4.3 GB）をロードして
合計 8.5 GB になっていた。chunk 125 の predict() でスパイクして OOM。

**修正** (`train_reranker.py` `_free_model`):

```python
def _free_model(model) -> None:
    try:
        if torch.cuda.is_available():
            model.to("cpu")          # ← 追加: VRAM ブロックを CPU に退避
            torch.cuda.synchronize()
    except Exception:
        pass
    del model
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
```

### フェーズ 2: 2 つ目の OOM（CPU RAM）

`_free_model` 修正後（v3）、chunk 125/125 採点完了・reranker 解放（VRAM=0.0 GB）まで
通過するようになったが、その後 `Dataset.from_dict` で CPU RAM OOM（23 GB）で終了。

```
dmesg: Out of memory: Killed process 35224 (python3) anon-rss:23271368kB
```

**原因**: `params.yaml` の `distill.num_negatives` が 7 のまま残っており、
3500 行 × 2 列（positive + negative PIL 画像）= 7000 枚の PIL 画像を
Arrow テーブルに一括エンコードして CPU RAM が 23 GB に達した。

また `_free_model` が内部で `del model` しても呼び出し元の変数参照が残るため、
CrossEncoder の 4 GB（CPU RAM、`model.to("cpu")` 済）が `Dataset.from_dict` 実行中に
まだ解放されていない可能性もあった。

**修正**:

1. `params.yaml` / `params_default.yaml` で `distill.num_negatives` を 7 → 3 に戻す
2. `distill.py` の `_teacher_reranker_scores` 返却直後に `gc.collect()` を追加:

```python
scores = _teacher_reranker_scores(cfg, personas, images, grouped)
gc.collect()   # CrossEncoder の CPU RAM（model.to("cpu")後）を確実に解放
rows = build_margin_rows(grouped, scores)
```

### 追加したデバッグログ（永続的に残す）

| ログ | 目的 |
|---|---|
| 採点チャンク完了（N/M: K スコア） | predict() 完了を確認 |
| reranker 解放後 VRAM | `_free_model` の効果を確認 |
| データセット構築完了 VRAM | Dataset.from_dict 後の状態 |
| student ロード完了 VRAM | 学習前 VRAM を確認 |

---

## その他の修正

### 採点パラメータの設定化

ハードコーディングを避け、`params.yaml` / CLI から制御可能にした。

| パラメータ | デフォルト | 説明 |
|---|---|---|
| `distill.score_chunk_size` | 32 | CrossEncoder 採点チャンクサイズ |
| `distill.score_batch_size` | 4 | CrossEncoder predict の batch_size |

---

## 実験設定

| パラメータ | Issue #25 | Issue #30 |
|---|---|---|
| `distill.num_negatives` | 7 | **3** |
| `distill.score_chunk_size` | 32 | 32（変更なし） |
| `distill.score_batch_size` | 8 → 4 | 4（変更なし） |
| `train.per_device_batch_size` | 2 | 2（変更なし） |
| `train.epochs` | 1 | 1（変更なし） |
| seed | 1 | 1（変更なし） |

4 variant は変更なし:

| variant | teacher | student 初期値 |
|---|---|---|
| `self` | reranker（FT済） | base 埋め込み |
| `ft_continue` | reranker（FT済） | FT 済み埋め込み |
| `oracle_base` | oracle（嗜好モデル soft label） | base 埋め込み |
| `oracle_ft` | oracle | FT 済み埋め込み |

---

## 結果

評価セット: 200 クエリ × 200 コーパス（7 ペルソナ × per-persona マクロ集計）。

### 埋め込み単体（bi-encoder のみ）

| モデル | MRR@10 | NDCG@10 | MAP@100 | Acc@1 | Recall@10 |
|---|---|---|---|---|---|
| base | 0.261 | 0.110 | 0.093 | 0.143 | 0.038 |
| finetuned | 0.655 | 0.597 | 0.470 | 0.571 | 0.194 |
| **distill_oracle_ft** | **0.806** | **0.634** | **0.485** | **0.714** | **0.214** |
| distill_oracle_base | 0.386 | 0.155 | 0.097 | 0.286 | 0.059 |
| distill_ft_continue | 0.386 | 0.130 | 0.084 | 0.286 | 0.031 |
| distill_self | 0.310 | 0.112 | 0.075 | 0.143 | 0.029 |

### リランク（6 パターン）

| 埋め込み | リランカー | MRR | NDCG@10 |
|---|---|---|---|
| ft | base | **0.786** | 0.594 |
| ft | ft | 0.750 | **0.601** |
| ft | none | 0.672 | 0.593 |
| base | none | 0.275 | 0.118 |
| base | base | 0.244 | 0.116 |
| base | ft | 0.187 | 0.106 |

### Issue #25（num_negatives=7）との比較

| モデル | MRR@10 (#25) | MRR@10 (#30) | 差分 |
|---|---|---|---|
| distill_oracle_ft | 0.810 | **0.806** | −0.004（ほぼ同等）|
| distill_oracle_base | 0.323 | **0.386** | +0.063 |
| distill_ft_continue | **0.697** | 0.386 | −0.311（大幅低下）|
| distill_self | **0.360** | 0.310 | −0.050 |

---

## 考察

### distill_oracle_ft が全モデル最高

`distill_oracle_ft`（MRR=0.806）は以下を上回る:

- **finetuned 単体** (0.655) より +0.151
- **ft + rerank=base** (0.786) より +0.020
- **ft + rerank=ft** (0.750) より +0.056

嗜好モデルの continuous soft label が FT 済み埋め込みを出発点に fine-tune することで、
リランカーなしで bi-encoder 単体が cross-encoder + リランクを超えるレベルに到達している。
`num_negatives=3` でも `num_negatives=7` と遜色ない（0.806 vs 0.810）ため、
oracle teacher は少ない負例で十分に機能する。

### reranker teacher は num_negatives への依存が高い

`distill_ft_continue`（reranker teacher）が 7→3 で MRR 0.697→0.386 と大幅低下。
MarginMSELoss は `s_pos − s_neg` のマージンを学習信号にするため、
多様な難負例からのマージン分布が重要。負例が 3 では十分な分散が得られない可能性がある。
num_negatives=7 に戻すには CPU RAM の制約を解決する必要がある
（現状: 16 GB RAM、7 negatives × 500 クエリ × 2 列の Arrow テーブルで 23 GB OOM）。

### VRAM: 学習中に 1168 MiB 退避

`distill_oracle_ft` / `distill_self` の学習でピーク VRAM 17548 MiB（物理 16380 MiB を超過）。
稼働はしているが共有メモリへの退避が発生している。`max_pixels` の縮小か
`per_device_batch_size=1` への変更で完全に収められる可能性がある。

---

## 今後の課題

| 課題 | 優先度 | 対応案 |
|---|---|---|
| reranker teacher を num_negatives=7 で安定動作させる | 中 | Dataset に PIL 画像でなくインデックスを格納し on-the-fly デコードに変更 |
| 学習中 VRAM スピル（+1168 MiB）を解消 | 低 | `max_pixels=100352` または `batch_size=1` に変更 |
| distill_oracle_ft の高精度をさらに検証 | 低 | seed 複数・eval 拡充で CI 幅を確認 |
