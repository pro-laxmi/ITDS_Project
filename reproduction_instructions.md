# Reproduction Instructions: Scalable Molecular Graph Retrieval

**Project:** Scalable Molecular Graph Retrieval (CS328 Final Project)  
**Author:** Laxmidhar Panda (Roll No: 24110185)  
**System Tested:** Linux (Fedora 44, Kernel 6.18+), Python 3.14.10, PyTorch 2.15.0+cu132, NVIDIA RTX 3050 Laptop GPU (6GB VRAM) / Intel Core i5 CPU  

---

## 1. Environment Setup

### 1.1 Create and Activate Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 1.2 Install Required Dependencies
```bash
pip install --upgrade pip

# Recommended: Install exact verified locked versions
pip install -r requirements-lock.txt

# Alternatively, manual installation:
# PyTorch (CUDA 12/13 or CPU):
pip install torch torchvision torchaudio
pip install torch_geometric
pip install rdkit faiss-cpu numpy scipy matplotlib scikit-learn pandas tqdm
```

---

## 2. Dataset Ingestion and Preprocessing

The ZINC 12,000-graph dataset is automatically downloaded into `data/ZINC` via PyTorch Geometric. Feature matrices and scaffold classes are cached to `data/processed_cache.pkl` (~16 MB).

To generate or rebuild the cache:
```bash
python -c "from src.data_ingestion import prepare_and_cache_dataset; prepare_and_cache_dataset()"
```

Output:
- Corpus size: 11,000 graphs (10k train + 1k val).
- Held-out test queries: 1,000 graphs.
- Cache file: `data/processed_cache.pkl`.

---

## 3. Training Graph Neural Network Models

All checkpoints are saved in `artifacts/`.

### 3.1 Train Standard GIN-SupCon Baseline
```bash
python src/train_gin.py
```
Outputs:
- Checkpoint: `artifacts/gin_best_model.pt`
- Embeddings: `artifacts/corpus_embeddings.npy` (11000, 256), `artifacts/query_embeddings.npy` (1000, 256)

### 3.2 Train Multi-Objective Models (GIN-CE, GIN-SupCon-Aug, GIN-ProxyAnchor)
```bash
python src/train_models.py
```
Outputs:
- GIN-CE: `artifacts/gin_ce_best.pt`, `artifacts/embs_ce_{corpus, query}.npy`
- GIN-SupCon-Aug: `artifacts/gin_supcon_aug_best.pt`, `artifacts/embs_supcon_aug_{corpus, query}.npy`
- GIN-ProxyAnchor: `artifacts/gin_proxyanchor_best.pt`, `artifacts/embs_proxyanchor_{corpus, query}.npy`

### 3.3 Training Label Strategy Comparison (A vs. B vs. C)
To train and evaluate the leak-free Strategy B (244 classes strictly from train) and Strategy C (610 classes including doubletons):
```bash
python src/train_strategy_comparison.py
```
Outputs:
- Serialized results: `artifacts/training_strategies_comparison.json`

### 3.4 Controlled Objective Benchmark under Leak-Free Strategy B
To train all five objective configurations (`GIN-CE`, `GIN-SupCon`, `GIN-SupCon-Aug`, `GIN-ProxyAnchor (Neg-Repeller)`, `GIN-ProxyAnchor (Excluded)`) under the common 244-class leak-free Strategy B taxonomy:
```bash
python src/train_strategy_b_objectives.py
```
Outputs:
- Checkpoints: `artifacts/gin_strat_b_{ce, supcon, supcon_aug, proxyanchor_neg, proxyanchor_excl}.pt`
- Serialized benchmark: `artifacts/strategy_b_objectives_comparison.json`

### 3.5 Multi-Seed Statistical Reliability Evaluation
To train and evaluate `GIN-ProxyAnchor (Strategy B, Excluded)` across independent random seeds (42, 43, 44):
```bash
python src/multiseed_strategy_b.py
```
Outputs:
- Multi-seed summary: `artifacts/multiseed_strategy_b_results.json`

---

## 4. Running the Multi-Depth Ablation Benchmark

### 4.1 Master Benchmark Structure and Pipeline Reconciliation
The complete evaluated master benchmark encompasses **18 pipeline configurations** (as detailed in `results_verified.csv` and Table 5 of `report_1.md`):
- **5 Discrete Baselines:** ECFP4 Shingle Jaccard, ECFP4 Tanimoto (1024-bit), MinHash LSH ($b=32, r=4$), MinHash LSH ($b=64, r=2$), Binary HNSW (Hamming).
- **10 Primary Leak-Free Strategy B Continuous Systems:** GIN-CE, GIN-SupCon, GIN-SupCon-Aug, GIN-ProxyAnchor (Neg-Repeller), and GIN-ProxyAnchor (Excluded), each evaluated on both exact FAISS Flat IP and FAISS HNSW ($efSearch=64$).
- **3 Diagnostic Systems:** Strategy A ProxyAnchor (Flat IP and HNSW) and Strategy C ProxyAnchor (HNSW).

To evaluate the baseline suite of 15 systems (5 discrete baselines + 10 continuous systems under legacy Strategy A embeddings):
```bash
python src/evaluate_exact_scaffolds.py
```
Output:
- Serialized JSON: `artifacts/exact_scaffold_evaluation.json`

To evaluate the 10 leak-free Strategy B systems:
```bash
python src/train_strategy_b_objectives.py
```
Output:
- Serialized JSON: `artifacts/strategy_b_objectives_comparison.json`

### 4.2 Legacy Secondary Experiment: Pseudo-Cluster Retrieval
To reproduce the legacy 336-class pseudo-cluster benchmark:
```bash
python src/ablation.py
```
Output:
- Serialized JSON: `artifacts/ablation_results.json`

---

## 5. Generating Publication-Quality Figures

Execute the four visualization scripts in `notebooks/`:

```bash
# 1. S-Curve Empirical vs. Theoretical Verification
python notebooks/01_eda_and_scurve.py
# Saved to: artifacts/s_curve_verification.png

# 2. Quality-Latency Pareto Frontier
python notebooks/02_pareto_plot.py
# Saved to: artifacts/pareto_frontier.png

# 3. Multi-Depth Curves (MAP@k and Recall@k for k in [1, 50])
python notebooks/03_depth_analysis.py
# Saved to: artifacts/depth_curves.png

# 4. Hyperparameter Sweeps (HNSW efSearch & MinHash Banding)
python notebooks/04_param_sweeps.py
# Saved to: artifacts/param_sweeps.png
```

---

## 6. Verification and Integrity Checkpoints

To ensure zero numerical fabrication:
1. **Hyperspherical Unit Normalization:**
   ```bash
   python -c "
   import numpy as np
   for f in ['corpus_embeddings.npy', 'embs_proxyanchor_corpus.npy', 'embs_strat_b_proxyanchor_excl_corpus.npy']:
       embs = np.load('artifacts/' + f)
       norms = np.linalg.norm(embs, axis=1)
       print(f'{f}: max dev = {np.max(np.abs(norms - 1.0)):.2e}')
   "
   ```
2. **ANN Recall Verification (Strategy B Primary Embeddings):**
   ```bash
   python -c "
   import faiss, numpy as np
   c = np.load('artifacts/embs_strat_b_proxyanchor_excl_corpus.npy')
   q = np.load('artifacts/embs_strat_b_proxyanchor_excl_query.npy')
   exact = faiss.IndexFlatIP(256); exact.add(c)
   hnsw = faiss.IndexHNSWFlat(256, 32, faiss.METRIC_INNER_PRODUCT); hnsw.hnsw.efSearch = 64; hnsw.add(c)
   _, e_top = exact.search(q, 10)
   _, h_top = hnsw.search(q, 10)
   rec = np.mean([len(set(e_top[i]) & set(h_top[i])) / 10.0 for i in range(len(q))])
   print(f'Primary Strategy B ProxyAnchor HNSW ANN Recall@10: {rec:.4f}')
   "
   ```
3. **Strategy B Objective Benchmark Integrity Check:**
   ```bash
   python -c "
   import json
   with open('artifacts/strategy_b_objectives_comparison.json') as f:
       res = json.load(f)
   for k, v in res.items():
       if 'HNSW' in k:
           print(f'{k:55s} | MAP@10: {v[\"covered\"][\"map_10\"]:.4f} | R@10: {v[\"covered\"][\"rec_10\"]:.4f} | ANN-Rec@10: {v[\"ann_recall_at_10\"]:.4f}')
   "
   ```
