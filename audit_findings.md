   # Scientific Audit Findings: Scalable Molecular Graph Retrieval (Final Verified Version)

   **Project:** Scalable Molecular Graph Retrieval: Sub-linear Structural Matching via Learned Hyperspherical Embeddings and Locality-Sensitive Indexing  
   **Course:** CS328: Introduction to Data Science, IIT Gandhinagar  
   **Student Author:** Laxmidhar Panda (Roll No: 24110185)  
   **Audit Date:** November 15, 2026 (Final Empirical Resolution & Methodological Sign-off)  
   **Status:** All Audit Directives Fully Executed, Empirically Reconciled, and Verified  

   ---

   ## 1. Executive Summary and Comprehensive Answers to the 10 Audit Questions

   Below is the definitive, evidence-backed resolution of each question posed in the independent scientific review:

   ### Question 1: Was training-label leakage present?
   **Answer: YES.**  
   *Evidence:* In `src/data_ingestion.py`, line 230, `all_scaffolds = corpus_scaffolds + query_scaffolds` pooled all 12,000 molecules (10k train + 1k val + 1k test) into `assign_structural_classes(min_class_size=3)`. An empirical audit of scaffold frequencies reveals that **exactly 62 non-empty scaffolds** had counts $< 3$ in the 10,000 training set (e.g. 1 or 2 instances), but were promoted to frequent supervised classes because instances in the validation and test sets pushed their count to $\ge 3$. Furthermore, 9,584 rare molecules were forced into 30 arbitrary hash pseudo-clusters (`overflow_offset + hash(scaffold) % 30`), introducing noisy positive supervision.

   ### Question 2: Were training labels changed?
   **Answer: YES, through controlled empirical ablations (Strategy A vs. Strategy B vs. Strategy C) and a complete leak-free multi-objective comparison.**  
   *Evidence:* To evaluate the exact impact of taxonomy design and eliminate data leakage, we implemented and evaluated:
   1. **Strategy A (Legacy Leaked 336-Class):** Baseline with 30 pseudo-clusters constructed by pooling all 12,000 molecules.
   2. **Strategy B (Train-Only 244-Class, Strict Leak-Free):** Scaffolds and frequencies derived strictly from the 10,000 training set (241 non-empty scaffolds $\ge 3$ + 3 acyclic bins = 244 classes). The 8,106 rare molecules are unclustered (label $-1$).
   3. **Strategy C (Train-Only 610-Class, Rare Expanded):** All scaffolds with $\ge 2$ instances in train receive a proxy ($241 + 366 + 3 = 610$ classes); singletons remain unclustered ($-1$).

   *Controlled Strategy B Multi-Objective Benchmark (Covered Scaffolds $N=275$):*
   - **GIN-CE (Strat B):** $\text{MAP}@10 = 0.2423$, $\text{Recall}@10 = \mathbf{0.1911}$, $\text{Acc}@10 = \mathbf{0.3055}$
   - **GIN-SupCon-Aug (Strat B):** $\text{MAP}@10 = 0.2928$, $\text{Recall}@10 = \mathbf{0.3663}$, $\text{Acc}@10 = \mathbf{0.3782}$
   - **GIN-ProxyAnchor (Strat B, Neg-Repeller):** $\text{MAP}@10 = 0.5418$, $\text{Recall}@10 = \mathbf{0.5127}$, $\text{Acc}@10 = \mathbf{0.5600}$
   - **GIN-ProxyAnchor (Strat B, Excluded):** $\text{MAP}@10 = \mathbf{0.5593}$, $\text{Recall}@10 = \mathbf{0.5244}$, $\text{Acc}@10 = \mathbf{0.5709}$
   - **GIN-SupCon (Strat B, Standard):** $\text{MAP}@10 = \mathbf{0.5996}$, $\text{Recall}@10 = \mathbf{0.5939}$, $\text{Acc}@10 = \mathbf{0.6073}$

   *Multi-Seed Statistical Reliability (Strategy B ProxyAnchor Excluded, Seeds 42, 43, 44):*
   - Individual Seed Runs: Seed 42 = $0.5457$, Seed 43 = $0.5673$, Seed 44 = $0.5547$ (Single-run primary benchmark = $0.5593$).
   - $\text{MAP}@10 = \mathbf{0.5559 \pm 0.0108}$ (sample std $s$, $N-1=2$; population std $\sigma = 0.0089$; range: $[0.5457, 0.5673]$)
   - $\text{Recall}@10 = \mathbf{0.5343 \pm 0.0135}$ (sample std $s$; population std $\sigma = 0.0111$; range: $[0.5209, 0.5480]$)
   - $\text{MRR} = \mathbf{0.7024 \pm 0.0092}$ (sample std $s$; population std $\sigma = 0.0075$; range: $[0.6960, 0.7129]$)
   - $\text{Acc}@10 = \mathbf{0.5636 \pm 0.0073}$ (sample std $s$; population std $\sigma = 0.0059$; range: $[0.5564, 0.5709]$)

   *Conclusion:* Even when all label leakage is eliminated and pseudo-clusters are completely removed, continuous representations under Strategy B achieve $\text{MAP}@10 = 0.5593$ (ProxyAnchor) and $0.5996$ (SupCon), outperforming exact ECFP4 Tanimoto ($0.2294$) by **$+143.8\%$ to $+161.4\%$ relative improvement**.

   ### Question 3: Was Acc@10 recomputed?
   **Answer: YES.**  
   *Evidence:* In `src/evaluate_exact_scaffolds.py` lines 51–57, majority-vote $\text{Acc}@10$ was recomputed strictly using **exact canonical Bemis-Murcko scaffold strings** (`MurckoScaffoldSmiles`), completely eliminating training labels $y$ and pseudo-clusters. Furthermore, rank tie-breaking was implemented deterministically: if multiple scaffolds achieve the same maximal vote in the top 10, the scaffold whose first occurrence appears at the highest retrieval rank (lowest index) is selected. On covered queries ($N=275$), `GIN-ProxyAnchor (Strat B)` achieves $\text{Acc}@10 = \mathbf{0.5709}$ (multi-seed: $\mathbf{0.5636 \pm 0.0073}$, sample std $s$; population std $\sigma = 0.0059$), compared to $0.3818$ for exact ECFP4 Tanimoto and $0.3055$ for GIN-CE. On all 1,000 queries, Strategy B ProxyAnchor achieves $\text{Acc}@10 = \mathbf{0.1550}$.

   ### Question 4: What are the corrected Pareto frontiers?
   **Answer: Both frontiers have been recalculated using exact unrounded values and delineated into two distinct operational perspectives:**
   1. **Index Search Latency Frontier (Primary Leak-Free Strategy B):**
      - `GIN-SupCon (Strategy B) + HNSW`: Search Latency = **$0.161\text{ ms}$**, $\text{MAP}@10 = \mathbf{0.5996}$
      *(Under Strategy A diagnostic: `GIN-ProxyAnchor (Strat A) + HNSW` achieves $0.161\text{ ms}$, $\text{MAP}=0.6227$)*.
   2. **End-to-End CPU Latency Frontier (Primary Leak-Free Strategy B):**
      - `ECFP4 + MinHash (b=32, r=4)`: E2E CPU Latency = **$0.477\text{ ms}$**, $\text{MAP}@10 = 0.1668$
      - `ECFP4 + Exact Tanimoto`: E2E CPU Latency = **$4.636\text{ ms}$**, $\text{MAP}@10 = 0.2294$
      - `GIN-SupCon (Strategy B) + HNSW`: E2E CPU Latency = **$10.036\text{ ms}$**, $\text{MAP}@10 = \mathbf{0.5996}$
      *(Under Strategy B ProxyAnchor: E2E CPU Latency = $10.036\text{ ms}$, $\text{MAP}=0.5593$)*.

   *Explanation:* The frontiers differ because GIN encoder CPU inference requires $9.88\text{ ms}$, accounting for $>98\%$ of total query runtime. In pure index search, GIN + HNSW dominates MinHash; in end-to-end CPU deployment, `ECFP4 + MinHash` ($0.477\text{ ms}$) provides an ultra-low latency non-neural baseline.

   ### Question 5: Why were the earlier timing measurements inconsistent?
   **Answer: Unbatched single-query dispatch vs. multi-threaded OpenMP BLAS matrix multiplication.**  
   *Evidence:* In the master benchmark, queries were timed in a single-query loop (`for q in range(1000): index.search(query[q:q+1], 10)`), measuring unbatched online request dispatch ($0.952\text{ ms}$ flat, $0.168\text{ ms}$ HNSW). In the scaling script, queries were submitted in a single batched call (`index.search(queries[:200], 10)`), triggering multi-threaded OpenMP GEMM (`cblas_sgemm`), which amortizes thread dispatch and reuses L3 cache, lowering per-query latency to $0.131\text{ ms}$ flat and $0.052\text{ ms}$ HNSW.

   ### Question 6: Were timings rerun under a common protocol?
   **Answer: YES.**  
   *Evidence:* Profiling on the Intel Core i5-13450HX CPU across all 1,000 queries established:
   - **Unbatched ($\text{Batch}=1$):** FAISS Flat IP Mean = $0.894\text{ ms}$ (Median: $0.861\text{ ms}$, P95: $1.035\text{ ms}$); FAISS HNSW Mean = $0.161\text{ ms}$ (Median: $0.165\text{ ms}$, P95: $0.208\text{ ms}$), achieving a **$5.56\times$ speedup**.
   - **Batched ($\text{Batch}=1000$):** FAISS Flat IP Mean = $0.137\text{ ms}$ / query (Median: $0.128\text{ ms}$, P95: $0.214\text{ ms}$); FAISS HNSW Mean = $0.022\text{ ms}$ / query (Median: $0.022\text{ ms}$, P95: $0.030\text{ ms}$), achieving a **$6.14\times$ speedup**.

   ### Question 7: Was molecular sanitization verified?
   **Answer: YES.**  
   *Evidence:* All 12,000 graphs from PyTorch Geometric ZINC were reconstructed and audited using strict `Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL)`. Exactly **12,000 / 12,000 molecules (100.00%) passed full sanitization with 0 errors**.

   ### Question 8: Did the main numerical results change?
   **Answer: The exact scaffold evaluation numbers are now fully anchored on the leak-free Strategy B taxonomy, with Strategy A retained as an explicit diagnostic across the unified 18-configuration master benchmark.**  
   *Evidence:* 
   - Across the unified master benchmark of **18 pipeline configurations** (5 static discrete + 10 primary Strategy B + 3 diagnostics in `results_verified.csv`), `GIN-SupCon (Strategy B)` attains the highest single-run peak retrieval performance on covered queries ($N=275$): Flat IP $\text{MAP}@10 = \mathbf{0.5996}$, $\text{Recall}@10 = \mathbf{0.5939}$, $\text{MRR} = \mathbf{0.7355}$, and $\text{Acc}@10 = \mathbf{0.6073}$.
   - `GIN-ProxyAnchor (Strategy B, Excluded)` achieves Flat IP $\text{MAP}@10 = \mathbf{0.5593}$, $\text{Recall}@10 = \mathbf{0.5244}$, $\text{MRR} = \mathbf{0.7091}$, and $\text{Acc}@10 = \mathbf{0.5709}$. FAISS HNSW ($efSearch=64$) achieves an identical $\text{MAP}@10 = \mathbf{0.5593}$ with an empirical **$100.00\%$ ANN Recall@10** on this 11,000-corpus configuration.
   - Across three independent training seeds (42, 43, 44), Strategy B ProxyAnchor exhibits high repeatability: $\text{MAP}@10 = \mathbf{0.5559 \pm 0.0108}$ (sample std $s$; population std $\sigma = 0.0089$).
   - Both continuous metric learning models dramatically surpass exact folded ECFP4 Tanimoto ($\text{MAP}@10 = 0.2294$) by $+143.8\%$ to $+161.4\%$, confirming that Hypothesis H1 holds rigorously without data leakage.
   - The legacy Strategy A result ($\text{MAP}@10 = 0.6227$) is explicitly identified as an optimistic diagnostic inflated by validation/test taxonomy leakage and 30 pseudo-clusters.

   ### Question 9: Which experiments remain single-seed?
   **Answer: GIN-CE, GIN-SupCon, and GIN-SupCon-Aug; GIN-ProxyAnchor was audited across multiple seeds.**  
   *Evidence:* 
   - To rigorously verify statistical stability, `GIN-ProxyAnchor (Strategy B)` was trained and evaluated across three independent random seeds (42, 43, 44). The resulting sample standard deviations were $s \approx 0.007 - 0.014$ ($\text{MAP}@10 = 0.5559 \pm 0.0108$, $\text{Recall}@10 = 0.5343 \pm 0.0135$, $\text{MRR} = 0.7024 \pm 0.0092$, $\text{Acc}@10 = 0.5636 \pm 0.0073$).
   - Other neural architectures (`gin_strat_b_ce.pt`, `gin_strat_b_supcon.pt`, `gin_strat_b_supcon_aug.pt`) were trained on a fixed seed (42) with deterministic PyTorch operations and validation loss checkpointing. Peak single-run performance belongs to GIN-SupCon ($0.5996$), whereas multi-seed confidence is established for GIN-ProxyAnchor ($0.5559 \pm 0.0108$).

   ### Question 10: Which limitations remain unresolved?
   **Answer:**
   1. **Corpus Coverage Constraint:** Exactly 72.5% of test queries possess zero exact scaffold matches in the 11,000-corpus due to the heavy-tailed singleton nature of the ZINC library.
   2. **GraphCL Augmentation Degradation:** Combining stochastic edge dropout (10%) and node masking (10%) reduced scaffold-level retrieval from $0.5996$ to $0.2928$. While a plausible hypothesis is that random bond dropping modifies ring frameworks and scaffold identities, confirming the exact causal factor would require isolating each perturbation operator via dedicated ablation experiments.
   3. **Negative Repulsion Artifacts in Metric Learning:** Naively treating unclustered rare molecules as negative repellers against all class proxies injects false-negative gradient signals, lowering $\text{MAP}@10$ from $0.5593$ to $0.5418$.
   4. **CPU Neural Inference Bottleneck:** Forward inference of a 4-layer GIN on CPU requires $9.88\text{ ms}$, representing $>98\%$ of total query time in single-query serving mode. Sub-linear ANN indexing accelerates vector comparison ($0.161\text{ ms}$), but neural feature extraction remains the primary operational bottleneck.

   ---

   ## 2. Dataset Partitioning and Census Verification

   Table 1: Audited Dataset Statistics and Structural Breakdown
   | Entity | Count | % of Total (12,000) | Notes |
   | :--- | :---: | :---: | :--- |
   | **Total Reconstructed Molecules** | 12,000 | 100.00% | 10,000 Train + 1,000 Val + 1,000 Test |
   | **Indexed Corpus ($\mathcal{C}$)** | 11,000 | 91.67% | Train $\cup$ Val |
   | **Query Set ($\mathcal{Q}$)** | 1,000 | 8.33% | Test split |
   | **Sanitization Failures** | **0** | **0.00%** | 100.00% sanitized cleanly |
   | **Acyclic Molecules (Empty Scaffold)** | 66 | 0.55% | 56 corpus, 10 queries |
   | **Non-Empty Ring Molecules** | 11,934 | 99.45% | Rings present |
   | **Unique Non-Empty Scaffolds** | 9,427 | — | Distinct Murcko SMILES |
   | **Singleton Scaffolds (Count = 1)** | 8,664 | 72.20% | Single-instance molecules |
   | **Doubleton Scaffolds (Count = 2)** | 460 | 7.67% | 920 molecules |
   | **Frequent Scaffolds (Count $\ge 3$)** | 303 | 19.58% | 2,350 molecules |
   | **Auxiliary Acyclic Pseudo-Classes** | 3 | 0.55% | 66 molecules (deterministic modulo-3 assignment) |
   | **Total Frequent Groups** | **306** | **20.13%** | $303 + 3 = 306$ |
   | **Overflow Rare Molecules** | **9,584** | **79.87%** | $8,664 + 920 = 9,584$ |
   | **Overflow Hash Pseudo-Clusters** | 30 | — | Rare molecules hashed modulo 30 |
   | **Total Supervised Training Labels** | **336** | **100.00%** | $\mathbf{306 + 30 = 336}$ |

   ---

   ## 3. Training Label Strategy Comparison Audit

   Table 2: Exact Murcko Retrieval Metrics under Training Label Taxonomy Ablation
   | Metric | Strategy A (Legacy Leaked 336-class) | Strategy B (Train-Only 244-class, Leak-Free) | Strategy C (Train-Only 610-class, Doubletons) |
   | :--- | :---: | :---: | :---: |
   | **Training Classes** | 336 (Pooled 12k) | 244 (Train only) | 610 (Train only) |
   | **Rare Molecule Handling** | 30 Hash Pseudo-Clusters | Unclustered Negatives ($-1$) | Unclustered Singletons ($-1$) |
   | **Covered MAP@10** | **0.6227** | **0.5471** | 0.4543 |
   | **Covered Recall@10** | **0.5706** | **0.5146** | 0.4379 |
   | **Covered Recall@50** | **0.7726** | **0.7230** | 0.6289 |
   | **Covered MRR** | **0.7478** | **0.6986** | 0.6218 |
   | **Covered Acc@10** | **0.6036** | **0.5636** | 0.4982 |
   | **All-1k MAP@10** | **0.1712** | **0.1505** | 0.1249 |
   | **All-1k Recall@10** | **0.1569** | **0.1415** | 0.1204 |
   | **All-1k Acc@10** | **0.1660** | **0.1550** | 0.1370 |

   *Origin of Strategy B 0.5471 in Table 2:* The value $0.5471$ in Table 2 stems from the preliminary pilot run in `src/train_strategy_comparison.py` (serialized in `artifacts/training_strategies_comparison.json`), which evaluated `ProxyAnchorWithUnlabeled` (Neg-Repeller). This exploratory run is retained in Table 2 as a historical comparison of initial taxonomy strategies, while Table 2b and the master benchmark (`results_verified.csv`) provide the authoritative controlled benchmark from `src/train_strategy_b_objectives.py` separating Neg-Repeller ($0.5418$), Excluded ($0.5593$), and SupCon ($0.5996$).

   ### 3.1 Controlled Objective Comparison under Common Leak-Free Strategy B Taxonomy
   All objectives were trained strictly on the 244 training-only classes without data leakage:

   Table 2b: Controlled Objective Benchmark (Strategy B, Covered Queries $N=275$)
   | Objective | Index Backend | Search Type | Covered MAP@10 | Covered R@10 | Covered R@50 | Covered MRR | Covered Acc@10 | ANN Recall@10 |
   | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
   | **GIN-CE** | FAISS IndexFlatIP | Exact | 0.2423 | 0.1911 | 0.3434 | 0.3659 | 0.3055 | 1.0000 |
   | **GIN-CE** | FAISS HNSW ($ef=64$) | Approx | 0.2423 | 0.1911 | 0.3434 | 0.3659 | 0.3055 | 1.0000 |
   | **GIN-SupCon-Aug** | FAISS IndexFlatIP | Exact | 0.2928 | 0.3663 | 0.5947 | 0.4760 | 0.3782 | 1.0000 |
   | **GIN-SupCon-Aug** | FAISS HNSW ($ef=64$) | Approx | 0.2928 | 0.3663 | 0.5947 | 0.4760 | 0.3782 | 1.0000 |
   | **GIN-ProxyAnchor (Neg)** | FAISS IndexFlatIP | Exact | 0.5418 | 0.5127 | 0.7179 | 0.6933 | 0.5600 | 1.0000 |
   | **GIN-ProxyAnchor (Neg)** | FAISS HNSW ($ef=64$) | Approx | 0.5418 | 0.5127 | 0.7179 | 0.6933 | 0.5600 | 1.0000 |
   | **GIN-ProxyAnchor (Excl)** | **FAISS IndexFlatIP** | **Exact** | **0.5593** | **0.5244** | **0.7168** | **0.7091** | **0.5709** | **1.0000** |
   | **GIN-ProxyAnchor (Excl)** | **FAISS HNSW ($ef=64$)** | **Approx** | **0.5593** | **0.5244** | **0.7168** | **0.7091** | **0.5709** | **1.0000** |
   | **GIN-SupCon (Standard)** | **FAISS IndexFlatIP** | **Exact** | **0.5996** | **0.5939** | **0.8001** | **0.7355** | **0.6073** | **1.0000** |
   | **GIN-SupCon (Standard)** | **FAISS HNSW ($ef=64$)** | **Approx** | **0.5996** | **0.5939** | **0.8001** | **0.7355** | **0.6073** | **1.0000** |

   > **Scientific Interpretation:** Under single-run evaluation, `GIN-SupCon (Standard)` achieves the highest peak retrieval metrics ($\text{MAP}@10 = \mathbf{0.5996}$). `GIN-ProxyAnchor (Excl)` delivers $\text{MAP}@10 = \mathbf{0.5593}$ in the primary run and possesses dedicated multi-seed statistical reliability across seeds 42, 43, 44 ($\text{MAP}@10 = \mathbf{0.5559 \pm 0.0108}$).

   ### 3.2 Multi-Seed Statistical Reliability (Strategy B ProxyAnchor)
   To verify reproducibility and rule out random initialization artifacts, `GIN-ProxyAnchor (Strategy B, Excluded)` was evaluated across three independent random seeds (42, 43, 44):

   Table 2c: Multi-Seed Evaluation of Strategy B ProxyAnchor (Exact Flat IP, $N=275$)
   | Seed | Covered MAP@10 | Covered Recall@10 | Covered R@50 | Covered MRR | Covered Acc@10 |
   | :---: | :---: | :---: | :---: | :---: | :---: |
   | **Seed 42** | 0.5457 | 0.5209 | 0.7268 | 0.6960 | 0.5564 |
   | **Seed 43** | 0.5673 | 0.5480 | 0.7393 | 0.7129 | 0.5636 |
   | **Seed 44** | 0.5547 | 0.5341 | 0.7324 | 0.6983 | 0.5709 |
   | **Mean $\pm$ Sample Std ($s$)** | **$0.5559 \pm 0.0108$** | **$0.5343 \pm 0.0135$** | **$0.7328 \pm 0.0062$** | **$0.7024 \pm 0.0092$** | **$0.5636 \pm 0.0073$** |
   | *(Population Std $\sigma$)* | *($\pm 0.0089$)* | *($\pm 0.0111$)* | *($\pm 0.0051$)* | *($\pm 0.0075$)* | *($\pm 0.0059$)* |

   *Seed-to-Checkpoint Mapping Note:* The primary single-run benchmark in Table 2b reports $\text{MAP}@10 = 0.5593$ from the saved checkpoint `gin_strat_b_proxyanchor_excl.pt`. The multi-seed run in `src/multiseed_strategy_b.py` evaluates three independently initialized training runs (seeds 42, 43, 44) yielding $\{0.5457, 0.5673, 0.5547\}$ with mean $0.5559$. The primary single-run value ($0.5593$) lies within the empirical $1s$ interval $[0.5451, 0.5667]$.

   ### 3.3 Mathematical Analysis of Label -1 and Negative Repulsion
   To verify that no objective introduces artificial positive grouping among rare molecules, we audited the exact implementation in `src/train_strategy_b_objectives.py`:
   - **GIN-CE (`train_ce_strategy_b`, lines 173–176):** Label $-1$ samples are filtered prior to dataset construction (`valid_indices = [i for i, lbl in enumerate(train_labels) if lbl >= 0]`). Unclustered molecules never enter cross-entropy loss or gradient updates.
   - **GIN-SupCon & GIN-SupCon-Aug (`train_supcon_strategy_b`, lines 211–215):** Label $-1$ samples are filtered prior to `ClassBalancedSampler(p_classes=16, k_instances=8)`. Unclustered molecules never enter contrastive batches or form pairs.
   - **GIN-ProxyAnchor Neg-Repeller (`ProxyAnchorWithUnlabeled`, lines 53–96):** In the forward pass, `valid_mask = (labels >= 0)` sets `one_hot = 0` for label $-1$. Unclustered molecules pull no positive proxy ($X_p^+ = \emptyset$), but interact with $(1.0 - \text{one\_hot}) = 1.0$ across all 244 class proxies, acting as universal negative repellers.
   - **GIN-ProxyAnchor Excluded (`ProxyAnchorExcludedUnlabeled`, lines 98–140):** Unlabeled samples are filtered dynamically within the forward pass (`features = features[valid_mask]`). They contribute zero loss and zero gradients.
   - **Universal Finding:** Across all five neural objectives, **label $-1$ is never treated as a positive class**, and unrelated rare molecules are never grouped as positive pairs.

   In deep metric learning with unclustered rare molecules ($y_i = -1$):
   1. **The Repeller Formulation (`ProxyAnchorWithUnlabeled`):**
      When unclustered molecules are included in training, they pull no positive proxy ($X_p^+ = \emptyset$), but they are assigned to $X_p^-$ for all $p \in \{1, \dots, C\}$. The gradient with respect to embedding $z_i$ repels every class proxy:
      $$\nabla_{z_i} \mathcal{L}_{\text{neg}} = \frac{1}{|P|} \sum_{p=1}^C \frac{e^{\alpha(z_i \cdot p + \delta)}}{1 + \sum_{j \in X_p^-} e^{\alpha(z_j \cdot p + \delta)}} \cdot \alpha p$$
      Notice the positive sign inside $e^{\alpha(z_i \cdot p + \delta)}$, matching standard Proxy-Anchor margin formulation. A supported mathematical interpretation of the observed performance drop ($0.5593 \to 0.5418$) is that forcing rare molecules to repel all proxies introduces conflicting gradient signals when rare molecules share local substructures (e.g. heterocyclic rings, linkers) with frequent classes.
   2. **The Exclusion Formulation (`ProxyAnchorExcluded`):**
      Completely filtering out unclustered molecules during loss calculation ($y_i = -1$ excluded) trains proxies purely on the 1,894 labeled instances, eliminating false-negative repulsion and raising $\text{MAP}@10$ to **$0.5593$** (multi-seed: $0.5559 \pm 0.0108$).
   3. **Training Data Utilization Trade-Off:**
      In Strategy B, exactly 1,894 of the 10,000 training molecules (18.94%) belong to the 244 frequent/acyclic classes and contribute supervised loss. The remaining 8,106 molecules (81.06%) are unclustered singletons and doubletons. While excluding them prevents gradient pollution, it also leaves rare scaffolds unrepresented during representation learning. Addressing this heavy-tailed distribution without arbitrary pseudo-clustering remains an open direction for self-supervised pretraining.

   ### 3.4 Statistical Audit of MinHash Affine Universal Hashing & SimHash Degeneration
   1. **MinHash Affine Permutations:**
      The 2-universal hash family $h_{a,b}(x) = (ax + b) \bmod p$ over 32-bit integer shingles guarantees pairwise universality ($\le 1/p$), but lacks full min-wise independence. Across 5 empirical seeds ($k=128$), the estimator exhibits an empirical downward bias ($\approx -0.02$ to $-0.05$) relative to exact Jaccard, while empirical standard deviations diverge visibly from the asymptotic bound $\sqrt{J(1-J)/k}$ (e.g. at $J=0.2$, $\sigma = 0.0167$ vs. $0.0354$). A 5-seed synthetic grid serves as a preliminary diagnostic rather than a general statistical proof.
   2. **SimHash Candidate Degeneration & Methodological Comparison:**
      - *Strategy B ProxyAnchor Evaluation:* Indexing Strategy B ProxyAnchor embeddings into `SimHashIndex(256, 128, 32, 4)` yields a single-query search latency of **$11.323\text{ ms}$** (median $10.370\text{ ms}$, P95 $17.492\text{ ms}$). While ANN Recall@10 is $1.0000$, logging reveals that SimHash pools an average of **$10,945$ candidates out of $11,000$ ($99.5\%$ of the database)**. On identical Strategy B embeddings, FAISS HNSW ($efSearch=64$) executes in **$0.161\text{ ms}$** with $1.0000$ ANN Recall@10, delivering a **$70.33\times$ search speedup** ($11.323\text{ ms} / 0.161\text{ ms}$).
      - *Legacy Strategy A GIN-SupCon Baseline:* In the historical diagnostic benchmark on Strategy A embeddings, SimHash executed in **$21.537\text{ ms}$** pooling $10,693$ candidates ($97.2\%$ of the database). Against that legacy baseline, HNSW ($0.168\text{ ms}$) achieved a **$128.20\times$ speedup** (or **$133.77\times$** compared against the unified $0.161\text{ ms}$ profile).
      - *Root Cause:* On $\mathbb{S}^{255}$, continuous embeddings cluster in a dense cone, causing random hyperplanes to partition poorly. Thus, SimHash degenerates into an unindexed linear scan, explaining its high latency ($11.3 - 21.5\text{ ms}$). Near-perfect ANN recall is an artifact of exhaustive candidate pooling, not effective indexing.

   ---

   ## 4. Serving Latency Breakdown and Disaggregation

   Table 3: Disaggregated Latency Profiling (Batch Size = 1)
   | Component | Device | Measured Latency | Operational Context |
   | :--- | :---: | :---: | :--- |
   | **RDKit ECFP4 Extraction** | CPU | $0.260\text{ ms}$ | $54.5\%$ of ECFP+MinHash |
   | **MinHash ($b=32, r=4$) Search** | CPU | $0.217\text{ ms}$ | $45.5\%$ of ECFP+MinHash |
   | **Total E2E (ECFP + MinHash)** | CPU | **$0.477\text{ ms}$** | Non-dominated baseline |
   | **ECFP4 + Exact Tanimoto Scan** | CPU | **$4.636\text{ ms}$** | Exact linear baseline |
   | **GIN 4-Layer Encoder Forward** | CPU | **$9.875\text{ ms}$** | $98.4\%$ of GIN+HNSW E2E |
   | **GIN 4-Layer Encoder Forward** | GPU (RTX 3050) | **$4.616\text{ ms}$** | GPU acceleration |
   | **GPU-to-CPU Embedding Transfer** | PCIe | **$0.062\text{ ms}$** | Minimal overhead |
   | **FAISS HNSW Vector Search ($ef=64$)**| CPU | **$0.161\text{ ms}$** | $1.6\%$ of GIN+HNSW E2E |
   | **Total E2E (GIN Strategy B + HNSW, CPU/CPU)** | CPU / CPU | **$10.036\text{ ms}$** | Dominates on quality |
   | **Total E2E (GIN Strategy B + HNSW, GPU/CPU)** | GPU / CPU | **$4.839\text{ ms}$** | Optimal quality-latency |

   ---

   ## 5. Summary of Visual Deliverables

   All figures have been generated and verified:
   1. `artifacts/pareto_frontier.png`: Combined 2-panel figure showing Index-Search Latency (Panel A) and End-to-End CPU Latency (Panel B).
   2. `artifacts/pareto_frontier_search.png`: Standalone high-resolution Search Latency Pareto plot.
   3. `artifacts/pareto_frontier_e2e.png`: Standalone high-resolution End-to-End CPU Latency Pareto plot.
   4. `artifacts/s_curve_verification.png`: Empirical MinHash S-curve validation against theoretical phase transition ($t_{50} = 0.3826$).
   5. `artifacts/depth_curves.png`: Precision and Recall curves across retrieval depths $k \in \{1, 5, 10, 50\}$.
   6. `artifacts/param_sweeps.png`: Parameter sensitivity sweeps for HNSW ($efSearch$) and MinHash ($b, r$).
