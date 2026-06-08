"""Qwen3-VL multimodal embedding fine-tuning demo.

Pipeline:
    1. generate_data  - synthesize a captioned image dataset with SD-Turbo
    2. evaluate       - measure text->image retrieval (NDCG / Recall@k)
    3. train          - fine-tune Qwen3-VL-Embedding-2B with Sentence Transformers
    4. rerank         - refine retrieved images with Qwen3-VL-Reranker-2B
"""

__version__ = "0.1.0"
