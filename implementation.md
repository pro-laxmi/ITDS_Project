# Implementation Specification: Advanced Molecular Graph Retrieval

**Course:** CS328 Introduction to Data Science  
**Student:** Laxmidhar Panda (Roll No: 24110185)  
**Project:** Scalable Molecular Graph Retrieval  
**Architecture:** 10-Pipeline Multi-Depth Benchmark (Discrete vs. Continuous $\times$ LSH vs. HNSW vs. Proxy-Anchor vs. Two-Stage Reranking)

---

## 1. Repository Structure

```text
molecular_retrieval/
├── data/                         # ZINC 12k dataset and processed caches
│   ├── ZINC/                     # Raw and subset PyG graph data
│   └── processed_cache.pkl       # 16 MB precomputed ECFP4 bitvectors, shingles, and labels
├── src/
│   ├── __init__.py               # Package marker
│   ├── data_ingestion.py         # PyG dataset loader, RDKit ECFP4, ClassBalancedSampler
│   ├── set_hashing.py            # Vectorized MinHash (k=128) and LSH Banding (b=32, r=4 & b=64, r=2)
│   ├── gin_encoder.py            # GIN with edge embeddings, GINClassifier, SupConLoss, ProxyAnchorLoss, GraphCL augs
│   ├── train_gin.py              # Single model training runner
│   ├── train_models.py           # Multi-model suite: trains GIN-CE, GIN-SupCon-Aug, GIN-ProxyAnchor
│   ├── vector_search.py          # SimHash, Continuous HNSW (with efSearch sweep), Binary HNSW, TwoStageRetriever
│   └── ablation.py               # Master 10-pipeline multi-depth benchmark harness
├── notebooks/                    # Analysis, verification, and publication visualization scripts
│   ├── 01_eda_and_scurve.py      # S-curve empirical validation vs. theoretical formula
│   ├── 02_pareto_plot.py         # 10-pipeline Quality-Latency Pareto frontier visualization
│   ├── 03_depth_analysis.py      # MAP@k and Recall@k curves across retrieval depth k in [1, 50]
│   └── 04_param_sweeps.py        # HNSW efSearch latency curve & MinHash banding threshold sweeps
├── artifacts/                    # Saved weights, benchmarks, JSON metrics, and figures
│   ├── gin_best_model.pt         # Trained GIN-SupCon checkpoint
│   ├── gin_ce_best.pt            # Trained GIN-CE checkpoint
│   ├── gin_supcon_aug_best.pt    # Trained GIN-SupCon-Aug checkpoint
│   ├── gin_proxyanchor_best.pt   # Trained GIN-ProxyAnchor checkpoint
│   ├── corpus_embeddings.npy     # Standard GIN-SupCon corpus embeddings (11000, 256)
│   ├── query_embeddings.npy      # Standard GIN-SupCon query embeddings (1000, 256)
│   ├── embs_ce_corpus.npy        # GIN-CE corpus embeddings (11000, 256)
│   ├── embs_ce_query.npy         # GIN-CE query embeddings (1000, 256)
│   ├── embs_supcon_aug_corpus.npy# GIN-SupCon-Aug corpus embeddings (11000, 256)
│   ├── embs_supcon_aug_query.npy # GIN-SupCon-Aug query embeddings (1000, 256)
│   ├── embs_proxyanchor_corpus.npy# GIN-ProxyAnchor corpus embeddings (11000, 256)
│   ├── embs_proxyanchor_query.npy # GIN-ProxyAnchor query embeddings (1000, 256)
│   ├── s_curve_verification.png  # Integrity check 1 plot (MAE = 0.0115)
│   ├── depth_curves.png          # Figure 3: MAP@k and Recall@k depth curves
│   ├── param_sweeps.png          # efSearch and Banding sensitivity plots
│   ├── pareto_frontier.png       # 10-pipeline Pareto frontier plot
│   └── ablation_results.json     # Master JSON metrics across all cutoffs and pipelines
├── requirements.txt              # Pinned environment dependencies
├── run_experiments.py            # Master execution script
├── theory.md                     # Comprehensive theoretical foundations
├── implementation.md             # This technical specification
└── report.md                     # Final course project report
```

---

## 2. Component Specifications

### 2.1 Component 1: `src/gin_encoder.py`
- **GraphCL Augmentations:**
  - `drop_edges(edge_index, edge_attr, p=0.1)`: Randomly drops 10% of edges.
  - `mask_features(x, p=0.1)`: Randomly masks 10% of atom tokens.
- **Model Architectures:**
  - `GINEncoder`: 4 layers of GINConv with edge projection, dual global readout (`global_add_pool` + `global_mean_pool`), projection head to 256-D, and $L_2$ normalization to $\mathbb{S}^{255}$.
  - `GINClassifier`: Encoder + linear classification head mapping 256-D to 336 classes for Cross-Entropy baseline.
- **Loss Functions:**
  - `SupConLoss`: Temperature $\tau = 0.07$, normalized multi-positive contrastive loss.
  - `ProxyAnchorLoss`: Parameterized class proxies $\boldsymbol{P} \in \mathbb{R}^{336 \times 256}$, margin $\delta = 0.1$, scaling factor $\alpha = 32.0$.

### 2.2 Component 2: `src/data_ingestion.py`
- **`ClassBalancedSampler`:**
  - Uniformly samples $P=16$ classes and $K=8$ instances per class per batch.
  - Guarantees exactly $K-1 = 7$ in-batch positive instances for every anchor.
- **Fingerprinting & Labeling:**
  - 12,000 ZINC molecules converted to RDKit `Chem.Mol` objects.
  - Sparse 32-bit integer shingles and 1024-bit folded bit vectors.
  - Ground-truth relevance via Bemis-Murcko scaffolds across 336 classes.

### 2.3 Component 3: `src/vector_search.py`
- **`ContinuousHNSWIndex`:**
  - FAISS HNSW flat with inner product metric, $M=32, efConstruction=200$.
  - Dynamic `set_ef_search(ef_search)` supporting sensitivity sweeps.
- **`SimHashIndex`:**
  - Hyperplane LSH with $\ell = 128$ random Gaussian projections.
  - Banding partitions ($b=32, r=4$).
- **`TwoStageRetriever`:**
  - Stage 1: Fast ANN candidate generation of top-$M$ shortlist ($M=50$) using HNSW.
  - Stage 2: Fine-grained exact metric reranking to produce final top-$K$.

### 2.4 Component 4: `src/ablation.py`
- Computes `MAP@1`, `MAP@5`, `MAP@10`, `MAP@50`, `Recall@1`, `Recall@5`, `Recall@10`, `Recall@50`, `MRR`, `Acc@10` (majority vote), and `Latency (ms)` across 10 pipelines:
  1. `ECFP + MinHash (b=32, r=4)`
  2. `ECFP + MinHash (b=64, r=2)`
  3. `ECFP + Binary HNSW`
  4. `GIN-CE + LSH`
  5. `GIN-CE + HNSW`
  6. `GIN-SupCon + LSH`
  7. `GIN-SupCon + HNSW`
  8. `GIN-SupCon + HNSW + Reranker`
  9. `GIN-SupCon-Aug + HNSW`
  10. `GIN-ProxyAnchor + HNSW`

---

## 3. Verification & Integrity Checkpoints

1. **Integrity Check 1 (S-Curve):** Verified empirical candidate collision rates against $P = 1 - (1 - s^r)^b$ ($N=4,000$ pairs, $\text{MAE} = 0.0115 \ll 0.05$).
2. **Integrity Check 2 (Hyperspherical Normalization):** Verified $\|\boldsymbol{z}\|_2 = 1.0 \pm 1.19 \times 10^{-7}$ for all corpus and query embeddings.
3. **Multi-Depth Verification:** Evaluated at depth $k=50$ proving that Recall climbs to $15.4\%$ (approaching the theoretical $17.0\%$ dataset limit).

---
*End of Implementation Specification.*
