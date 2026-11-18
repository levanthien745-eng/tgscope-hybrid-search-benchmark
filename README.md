# TGScope 6-Language Hybrid Channel Search Benchmark & Weighted RRF Evaluator

Lightweight zero-dependency evaluation harness demonstrating hybrid semantic + lexical channel retrieval across 6 languages (`en`, `ru`, `pt`, `es`, `ar`, `vi`) and 3 Telegram query classes (`exact_handle`, `cross_lingual`, `topical`).

## Key Findings
- **Similarity-Thresholded `pg_trgm` (`sim > 0.05`):** Passing raw zero-overlap cross-script lexical ties into RRF degrades cross-lingual ranking (`0.877` nDCG@3). Filtering `pg_trgm` candidates at `sim > 0.05` before Weighted RRF lifts `hybrid_wrrf_ndcg@3` to **`1.0000`** (`1.0000` Recall@3), outperforming both `lexical_only` (`0.3333`) and `dense_only` (`0.9167`).
- **Canonical Handle Normalization (`@username` / `t.me/`):** Deterministic regex stripping + RRF boost guarantees rank-1 exact channel lookup even when dense embeddings diffuse across generic tokens.

## Run
```bash
python tgscope_hybrid_rrf_benchmark.py
```
