# 実験レポート: 知識蒸留（teacher→student 埋め込み）

**実施日**: 2026-06-18  
**ブランチ**: `claude/modest-heisenberg-136hps`  
**関連 Issue / PR**: Issue #19 / PR #18

---

## 1. 背景

埋め込み FT（`train.py` / MNRL）とリランカー FT（`train_reranker.py` / BCE）に続く 3 本目の学習軸として、**知識蒸留**（`distill.py`）を追加した。

preference タスク（Issue #15）により「ペルソナの嗜好に基づく非加法的な交互作用」が再現できるようになったが、その賢さはリランカー（cross-encoder）に閉じており、内積で高速に動く第 1 段（bi-encoder）には反映されていない。本実験では、リランカーや嗖好モデルの知識を bi-encoder に蒸留することで第 1 段の精度が向上するかを検証した。

---

## 2. 実験設定

### 2.1 パイプライン構成

```
generate_data → train → train_reranker → distill → eval_distill
```

### 2.2 データセット

| 項目 | 値 |
|------|----|
| 生成画像数（train） | 500 枚 |
| 評価画像数（eval） | 200 枚 |
| タスク | preference（ペルソナ嗜好モデル, gamma=2.0） |
| 画像生成モデル | FLUX.2-klein-4B（4 step） |

### 2.3 蒸留設定

| パラメータ | 値 |
|----------|----|
| teacher（variant: self / ft_continue） | reranker（FT 済みリランカー） |
| 損失関数 | MarginMSELoss（リランカーの score_pos − score_neg を student に転写） |
| student（self） | ベース埋め込みモデル（Qwen3-VL-Embedding-2B）から学習 |
| student（ft_continue） | FT 済み埋め込みモデルから継続学習 |
| num_negatives | 3（ハードネガティブマイニング） |
| epochs | 1 |
| batch_size | 2 |
| learning_rate | 2e-5 |
| gradient_checkpointing | true |
| save_steps | 10000（実質 checkpoint なし） |

### 2.4 VRAM 使用状況

学習中（student 単体）に dedicated 約 16 GB・shared 約 6 GB の spill が発生したが、
モデルは埋め込み → リランカー → student の順に 1 つずつロード・解放しており同時同居はなかった。
spill による速度劣化は観測されず、学習速度は step あたり約 7.7 秒で安定していた。

---

## 3. 結果

評価セット 200 クエリ × 200 コーパス、クエリごとの正解画像は 50 枚（N:N relevance）。

| モデル | NDCG@10 | MRR@10 | Recall@1 | Recall@5 | Recall@10 |
|--------|---------|--------|----------|----------|-----------|
| base（ベースライン） | 0.1335 | 0.2735 | 0.0050 | 0.0100 | 0.0350 |
| finetuned（埋め込み FT） | **0.5579** | **0.5617** | 0.0050 | 0.0750 | 0.1750 |
| distill_self（base→蒸留） | 0.1271 | 0.2232 | 0.0000 | 0.0197 | 0.0433 |
| distill_ft_continue（FT→蒸留） | 0.1435 | 0.3425 | 0.0023 | 0.0322 | 0.0452 |

---

## 4. 考察

### 4.1 distill_self：base を下回り、蒸留の効果が出なかった

`distill_self`（base から蒸留）は NDCG@10=0.1271 と base（0.1335）をわずかに下回った。
MarginMSE の損失値が学習全体を通して 100〜200 台と非常に大きく（cf. oracle 蒸留の smoke では 0.1〜2 程度）、teacher スコア（リランカー logit）と student スコア（内積）のスケール差が未補正なまま学習したことが主因と考えられる。結果として student の表現が意味ある方向に誘導されなかった可能性が高い。

### 4.2 distill_ft_continue：self よりは良いが改善は軽微

`distill_ft_continue`（FT 済み埋め込みから継続蒸留）は NDCG@10=0.1435、MRR@10=0.3425 と base を上回り、distill_self より高い。

FT 済みモデルはすでに良好な表現を持っているため、MarginMSE の歪んだ学習信号に対してもある程度耐性があったと解釈できる。ただし finetuned（NDCG@10=0.558）との差は依然大きく、蒸留による実質的な上乗せ効果はほぼない。

### 4.3 実行間での結果のばらつき

前回の手動実行（同一ブランチ）と今回の `dvc repro` では数値が異なる。データセット自体は同一（テキストは `random.Random(seed=42)` 固定、画像は `.cache/imggen` キャッシュから同一ファイルを使用）だが、GPU 上の演算（Flash Attention, cuBLAS など）は同一シードでも厳密な再現性を保証しないため、学習結果にブレが生じる。

finetuned の NDCG@10 が 0.647 → 0.558 と約 0.09 振れており、このブレが distill_self と base の大小逆転を引き起こしている。少データ・短 epoch の設定下では学習の確率的なブレへの感度が高く、複数 seed での平均化や評価セットの増量が安定化に有効と考えられる。

### 4.4 蒸留 teacher のスケール差（根本課題）

MarginMSE は `s_pos − s_neg`（リランカーの logit 差）を student の内積差に合わせる損失だが、両者のスケールが大きく異なる。teacher スコアの正規化（sigmoid / z-score）または損失重みのスケーリングが最優先の改善候補。

---

## 5. まとめ

| 受け入れ条件 | 結果 |
|-------------|------|
| GPU で distill@self 完走 | ✓ |
| GPU で distill@ft_continue 完走 | ✓ |
| make smoke（CPU, teacher=oracle）完走 | ✓ |
| 蒸留後 student が base より改善 | △（distill_ft_continue のみ、軽微） |
| VRAM モデル同居なし | ✓ |

- **蒸留パイプラインとして動作する**ことは確認できた
- teacher スコアと student スコアのスケール差が未補正で、MarginMSE の学習信号が機能していない可能性が高い
- distill_ft_continue が distill_self をわずかに上回るが、finetuned との差は大きく実用的な改善には至っていない
- データセットの乱数依存性が高く、結果の安定性に課題がある

---

## 6. 今後の改善候補

1. **teacher スコアの正規化**: リランカー logit に sigmoid や z-score を適用して student の内積スケールに合わせる
2. **distill_ft_continue の学習率低減**: FT 済み表現を壊さないよう learning_rate を 1e-6 程度に下げる
3. **oracle 蒸留の本番実行**: 今回は teacher=reranker のみ。teacher=oracle（嗖好モデルの連続 appeal）との比較が未実施
4. **epoch 数増加**: 1 epoch では収束しきれていない可能性（eval NDCG が学習後半も改善傾向）
