"""Qwen3-VL マルチモーダル埋め込み ファインチューニング・デモ パッケージ。

このパッケージは「合成データだけで画像検索の精度を上げる」一連の流れを、
4 つのステップに分けて提供する。各ステップは独立したモジュールになっており、
``python -m qwen3vl_demo.<module>`` で個別に実行できる（Makefile / DVC からも呼ばれる）。

パイプライン全体像::

    1. generate_data  キャプションから FLUX.2-klein で画像を生成し、ペルソナ嗜好で
                      自動ラベル付けしたデータセットを作る（ペルソナ名＝検索クエリ）
    2. evaluate       ペルソナ→画像検索の精度（NDCG / Recall@k など）を測定する
    3. train          Qwen3-VL-Embedding-2B を Sentence Transformers で微調整する
    4. rerank         検索の上位候補を Qwen3-VL-Reranker-2B で並べ替えて仕上げる

各モジュールの責務と相互依存については ``docs/architecture.md`` を参照。
"""

# パッケージのバージョン。pyproject.toml の version と一致させること。
__version__ = "0.1.0"
