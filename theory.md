# Theoretical Foundations of Scalable Molecular Graph Retrieval

**Course:** CS328: Introduction to Data Science  
**Author:** Laxmidhar Panda (Roll No: 24110185)  
**Project:** Scalable Molecular Graph Retrieval  
**Architecture:** Discrete Set Hashing vs. Learned Hyperspherical Geometric Spaces on the ZINC Dataset

---

## Table of Contents
1. [The Molecular Retrieval Problem & Computational Complexity](#1-the-molecular-retrieval-problem--computational-complexity)
   - 1.1 Molecules as Attributed Graphs
   - 1.2 Exact Subgraph Isomorphism & NP-Completeness
   - 1.3 Graph Edit Distance (GED)
   - 1.4 The Search Scalability Bottleneck
2. [Discrete Chemoinformatics & Fingerprints](#2-discrete-chemoinformatics--fingerprints)
   - 2.1 The Extended-Connectivity Fingerprint (ECFP / Morgan Algorithm)
   - 2.2 Radial Expansion and Neighborhood Invariants
   - 2.3 Folding and Hashing Shingles into Fixed-Width Vectors
   - 2.4 Jaccard & Tanimoto Similarity Metric Space
3. [Locality-Sensitive Hashing (LSH) for Discrete Sets](#3-locality-sensitive-hashing-lsh-for-discrete-sets)
   - 3.1 Min-wise Independent Permutations (MinHash)
   - 3.2 Proof of MinHash Collision Identity
   - 3.3 Universal Hashing Implementation
   - 3.4 Banding Architecture ($k = b \cdot r$)
   - 3.5 Derivation and Analysis of the S-Curve Phase Transition
4. [Neural Representation Learning: Graph Isomorphism Networks (GIN)](#4-neural-representation-learning-graph-isomorphism-networks-gin)
   - 4.1 Message Passing Neural Networks (MPNN) and Expressive Limits
   - 4.2 The 1-Weisfeiler-Lehman (1-WL) Graph Isomorphism Test
   - 4.3 Xu et al. Theorem: Injective Multiset Aggregation
   - 4.4 The GIN Layer Formulation with Bond/Edge Attributes
   - 4.5 Global Readout: Sum vs. Mean Pooling (Atom Count vs. Structural Density)
   - 4.6 Hyperspherical Normalization: $L_2$ Projection onto $\mathbb{S}^{d-1}$
5. [Metric Learning Objectives & Sampling Strategies](#5-metric-learning-objectives--sampling-strategies)
   - 5.1 Cross-Entropy vs. Metric-Aligned Geometry
   - 5.2 Supervised Contrastive Loss (SupCon) & Temperature Sensitivity
   - 5.3 GraphCL Stochastic View Augmentations (Edge Dropout & Feature Masking)
   - 5.4 The Power-Law Class Imbalance Problem & Class-Balanced Sampling
   - 5.5 Proxy-Anchor Loss: Deep Metric Learning via Class Proxies
6. [Sublinear Vector Serving in Continuous Metric Spaces](#6-sublinear-vector-serving-in-continuous-metric-spaces)
   - 6.1 The Curse of Dimensionality in $\mathbb{R}^d$
   - 6.2 Hyperplane Locality-Sensitive Hashing (SimHash)
   - 6.3 Angular Distance and Collision Probability Proof
   - 6.4 Hierarchical Navigable Small World (HNSW) Graphs
   - 6.5 Multi-Layer Skip-List Graph Search
   - 6.6 HNSW Construction and Search Hyperparameters ($M$, $efConstruction$, $efSearch$)
   - 6.7 Two-Stage Retrieval: Fast ANN Candidate Shortlisting + Exact Reranking
7. [The Multi-Depth Factorial Ablation Matrix & Retrieval Metrics](#7-the-multi-depth-factorial-ablation-matrix--retrieval-metrics)
   - 7.1 Multi-Depth Evaluation Protocol: MAP@k, Recall@k, MRR, Acc@k
   - 7.2 The Mathematical "Recall@10 Ceiling"
   - 7.3 Latency Profiling and Empirical Complexity
   - 7.4 Quality-Latency Pareto Frontier Optimization

---

## 1. The Molecular Retrieval Problem & Computational Complexity

### 1.1 Molecules as Attributed Graphs
A molecule is naturally represented as an undirected attributed graph:
$$\mathcal{G} = (\mathcal{V}, \mathcal{E}, \boldsymbol{X}_v, \boldsymbol{E}_{uv})$$
where:
- $\mathcal{V} = \{v_1, v_2, \dots, v_{|\mathcal{V}|}\}$ is the set of vertices representing atoms.
- $\mathcal{E} \subseteq \mathcal{V} \times \mathcal{V}$ is the set of edges representing covalent chemical bonds.
- $\boldsymbol{X}_v \in \mathbb{R}^{|\mathcal{V}| \times d_v}$ is the node feature matrix capturing atomic attributes:
  - Atomic number $Z$ (e.g., C=6, N=7, O=8, S=16, Halogens).
  - Formal charge, hybridization state ($sp, sp^2, sp^3$), aromaticity flag, hydrogen count (explicit/implicit), and chirality.
- $\boldsymbol{E}_{uv} \in \mathbb{R}^{|\mathcal{E}| \times d_e}$ is the edge feature matrix capturing bond attributes:
  - Bond order (single, double, triple, aromatic).
  - Stereochemical configuration ($E/Z$, cis/trans) and conjugation status.

### 1.2 Exact Subgraph Isomorphism & NP-Completeness
Given a query molecular graph $\mathcal{G}_q = (\mathcal{V}_q, \mathcal{E}_q)$ and a target database graph $\mathcal{G}_t = (\mathcal{V}_t, \mathcal{E}_t)$, the **subgraph isomorphism problem** seeks an injective mapping $f: \mathcal{V}_q \to \mathcal{V}_t$ such that:
1. $\forall u \in \mathcal{V}_q, \boldsymbol{X}_q(u) = \boldsymbol{X}_t(f(u))$ (matching node attributes).
2. $\forall (u, v) \in \mathcal{E}_q, (f(u), f(v)) \in \mathcal{E}_t$ and $\boldsymbol{E}_q(u, v) = \boldsymbol{E}_t(f(u), f(v))$ (matching edge attributes).

- Subgraph isomorphism is proven **NP-complete** (Cook, 1971).
- Classic exact algorithms like **Ullmann's Algorithm** (1976) and **VF2** (Cordella et al., 2004) use state-space tree search with depth-first search and consistency pruning.
- In the worst-case scenario: $\mathcal{O}(|\mathcal{V}_t|^{|\mathcal{V}_q|} \cdot |\mathcal{V}_q|)$.
- Even with chemical heuristic pruning, checking a single query against a database of $N = 10^6$ to $10^9$ molecules requires exhaustive pair-wise matching, which is completely computationally infeasible in real-time screening.

### 1.3 Graph Edit Distance (GED)
When looking for *similar* rather than exact subgraphs, the standard mathematical formulation is the **Graph Edit Distance (GED)**:
$$\operatorname{GED}(\mathcal{G}_1, \mathcal{G}_2) = \min_{(e_1, \dots, e_k) \in \Upsilon(\mathcal{G}_1, \mathcal{G}_2)} \sum_{i=1}^k c(e_i)$$
where $\Upsilon(\mathcal{G}_1, \mathcal{G}_2)$ is the set of all edit paths transforming $\mathcal{G}_1$ into $\mathcal{G}_2$ using basic edit operations (vertex insertion/deletion/substitution, edge insertion/deletion/substitution), each having non-negative cost $c(e_i)$.
- Computing exact GED is **NP-hard**, and approximating it remains APX-hard.

### 1.4 The Search Scalability Bottleneck
To achieve real-time molecular discovery, chemical search systems must satisfy two criteria:
1. **Constant or Sublinear Query Time:** Finding top-$K$ candidates in $\mathcal{O}(\log N)$ or $\mathcal{O}(1)$ time with respect to corpus size $N$.
2. **High Semantic Fidelity:** The retrieved candidates must preserve genuine chemical reactivity, bioisosteric equivalence, and structural pharmacophore properties.

---

## 2. Discrete Chemoinformatics & Fingerprints

### 2.1 The Extended-Connectivity Fingerprint (ECFP / Morgan Algorithm)
The standard chemoinformatics baseline is the **Extended-Connectivity Fingerprint (ECFP)** (Rogers & Hahn, 2010), rooted in the Morgan algorithm (1965) for molecular canonicalization. ECFP fingerprints capture circular atom neighborhoods up to diameter $2r$ (e.g. ECFP4 corresponds to radius $r = 2$).

### 2.2 Radial Expansion and Neighborhood Invariants
1. **Stage 0: Initialization of Atom Identifiers**
   $$h_v^{(0)} = \operatorname{Hash}\Big(\text{AtomicNumber}(v), \text{HeavyDegree}(v), \text{Valence}(v), \text{Charge}(v), \text{Aromaticity}(v)\Big)$$
2. **Stage 1 to $r$: Radial Expansion**
   $$T_v^{(t)} = \Big\{ \big(\text{BondOrder}(u, v), h_u^{(t-1)}\big) \;\Big|\; u \in \mathcal{N}(v) \Big\}$$
   $$h_v^{(t)} = \operatorname{Hash}\Big( h_v^{(t-1)}, \operatorname{sort}\big(T_v^{(t)}\big) \Big)$$
3. **Stage 3: Shingle Set Collection**
   $$\mathcal{S}(\mathcal{G}) = \bigcup_{t=0}^r \bigcup_{v \in \mathcal{V}} \big\{ h_v^{(t)} \big\}$$
   Each unique 32-bit integer in $\mathcal{S}(\mathcal{G})$ represents a distinct circular substructure (a "shingle").

### 2.3 Folding and Jaccard / Tanimoto Metric Space
To convert the unbounded sparse integer set $\mathcal{S}(\mathcal{G})$ into a dense binary bit vector $\boldsymbol{x} \in \{0, 1\}^D$ (e.g., $D = 1024$):
$$x_j = \mathbb{I}\Big(\exists s \in \mathcal{S}(\mathcal{G}) \text{ s.t. } (s \bmod D) = j\Big)$$
For shingle sets $A = \mathcal{S}(\mathcal{G}_1)$ and $B = \mathcal{S}(\mathcal{G}_2)$, the **Jaccard similarity** is:
$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$
For binary folded vectors $\boldsymbol{u}, \boldsymbol{v} \in \{0, 1\}^D$, the equivalent **Tanimoto coefficient** is:
$$T(\boldsymbol{u}, \boldsymbol{v}) = \frac{\boldsymbol{u} \cdot \boldsymbol{v}}{\|\boldsymbol{u}\|_1 + \|\boldsymbol{v}\|_1 - \boldsymbol{u} \cdot \boldsymbol{v}} = \frac{c}{a + b - c}$$

---

## 3. Locality-Sensitive Hashing (LSH) for Discrete Sets

### 3.1 Min-wise Independent Permutations (MinHash)
Let $\pi: \mathcal{U} \to \mathcal{U}$ be a random permutation uniformly chosen from the symmetric group on the shingle universe $\mathcal{U}$. The **MinHash** function is:
$$h_\pi(A) = \min_{x \in A} \pi(x)$$

### 3.2 Proof of MinHash Collision Identity
**Theorem:** For any two sets $A, B \subseteq \mathcal{U}$:
$$\Pr_{\pi}\Big[h_\pi(A) = h_\pi(B)\Big] = J(A, B)$$
**Proof:**
Consider $C = A \cup B$. The random permutation $\pi$ places every element of $C$ in a totally ordered rank with equal probability $\frac{1}{|C|} = \frac{1}{|A \cup B|}$. The event $h_\pi(A) = h_\pi(B)$ occurs if and only if the element that minimizes $\pi$ over $A \cup B$ falls within the intersection $A \cap B$. If the minimizer is in $A \setminus B$, then $h_\pi(A) < h_\pi(B)$; if in $B \setminus A$, then $h_\pi(B) < h_\pi(A)$. Thus:
$$\Pr\big[h_\pi(A) = h_\pi(B)\big] = \frac{|A \cap B|}{|A \cup B|} = J(A, B) \quad \blacksquare$$

### 3.3 Universal Hashing Implementation
We approximate random permutations using $k$ universal hash functions:
$$h_i(x) = (a_i \cdot x + b_i) \bmod p, \quad p = 4294967311 > 2^{32}$$

### 3.4 Banding Architecture & The S-Curve Phase Transition
Partition the $k$ signature rows into $b$ bands of $r$ rows ($k = b \cdot r$).
1. Probability of all $r$ rows matching in a single band: $P = s^r$ (where $s = J(A, B)$).
2. Probability of a mismatch in a single band: $1 - s^r$.
3. Probability that all $b$ bands fail to match: $(1 - s^r)^b$.
4. **Candidate Selection Probability:**
   $$P_{\text{cand}}(s) = 1 - (1 - s^r)^b$$
- **Inflection Point Threshold:**
   $$t^* \approx \left(\frac{1}{b}\right)^{1/r}$$
   - For $b=32, r=4$: $t^* \approx (1/32)^{0.25} \approx 0.4204$ (Balanced threshold).
   - For $b=64, r=2$: $t^* \approx (1/64)^{0.5} = 0.1250$ (Relaxed threshold, higher recall).
   - For $b=16, r=8$: $t^* \approx (1/16)^{0.125} \approx 0.7071$ (Strict threshold, higher precision).

---

## 4. Neural Representation Learning: Graph Isomorphism Networks (GIN)

### 4.1 Message Passing Neural Networks (MPNN) and Expressive Limits
Standard MPNNs (GCN, GraphSAGE) use mean or max pooling, which cannot distinguish simple non-isomorphic multiset neighborhoods (Xu et al., ICLR 2019). Mean aggregation captures feature proportions but loses scale/multiplicity; max aggregation identifies presence but ignores counts.

### 4.2 The 1-WL Test and Multiset Injectivity
**Theorem (Xu et al., ICLR 2019):**
*A message-passing GNN achieves maximal 1-Weisfeiler-Lehman (1-WL) discriminative power if and only if its neighborhood aggregation and graph readout functions are injective on multisets.*

### 4.3 The GIN Layer Formulation with Bond Attributes
To achieve injectivity while injecting categorical bond types $e_{uv}$:
$$h_v^{(l)} = \operatorname{MLP}^{(l)}\left( (1 + \epsilon^{(l)}) h_v^{(l-1)} + \sum_{u \in \mathcal{N}(v)} \operatorname{ReLU}\big( h_u^{(l-1)} + \boldsymbol{W}_e^{(l)} \boldsymbol{e}_{uv} \big) \right)$$
where $\operatorname{MLP}^{(l)}$ is a 2-layer perceptron with Batch Normalization and GELU activation.

### 4.4 Dual Global Readout Pooling
To capture both extensive properties (molecular size, atom counts) and intensive properties (density, scaffold geometry):
$$\boldsymbol{h}_{\mathcal{G}} = \Big[ \operatorname{global\_add\_pool}(h^{(4)}, \text{batch}) \;\big\|\; \operatorname{global\_mean\_pool}(h^{(4)}, \text{batch}) \Big] \in \mathbb{R}^{2 \cdot d_{\text{hidden}}}$$

### 4.5 Hyperspherical Normalization: $L_2$ Projection onto $\mathbb{S}^{d-1}$
The projection head maps $\boldsymbol{h}_{\mathcal{G}}$ to $\mathbb{R}^{256}$, normalized to the unit hypersphere:
$$\boldsymbol{z} = \frac{\boldsymbol{z}_{\text{proj}}}{\|\boldsymbol{z}_{\text{proj}}\|_2} \in \mathbb{S}^{255}$$
On $\mathbb{S}^{255}$, Euclidean distance and Inner Product are strictly equivalent:
$$\|\boldsymbol{z}_i - \boldsymbol{z}_j\|_2^2 = 2 - 2 \langle \boldsymbol{z}_i, \boldsymbol{z}_j \rangle$$

---

## 5. Metric Learning Objectives & Sampling Strategies

### 5.1 Cross-Entropy vs. Metric-Aligned Geometry
Standard cross-entropy (`GIN-CE`) optimizes a linear classifier atop embeddings. While it achieves high classification accuracy, it optimizes for separating hyperplanes rather than intra-class compactness and metric distance. Contrastive and metric learning objectives explicitly reshape the hypersphere geometry for nearest-neighbor retrieval.

### 5.2 Supervised Contrastive Loss (SupCon)
$$\mathcal{L}_{\text{SupCon}} = \sum_{i \in I} \frac{-1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(\boldsymbol{z}_i \cdot \boldsymbol{z}_p / \tau)}{\sum_{a \in A(i)} \exp(\boldsymbol{z}_i \cdot \boldsymbol{z}_a / \tau)}$$
- Temperature $\tau = 0.07$ acts as a hardness parameter: smaller $\tau$ penalizes hard negative impostors with higher gradient magnitude.

### 5.3 GraphCL Stochastic View Augmentations
To prevent representation collapse and improve generalization, we apply stochastic graph augmentations (You et al., NeurIPS 2020):
1. **Edge Dropout:** $\mathcal{E}' \subset \mathcal{E}$ randomly removes bonds with probability $p=0.1$, forcing the encoder to learn robust topological motifs.
2. **Feature Masking:** Atom tokens are masked with probability $p=0.1$, requiring the model to infer missing chemical identity from context.

### 5.4 Class-Balanced Mini-Batch Sampling
In molecular datasets, scaffold frequencies follow a power-law distribution. In standard random mini-batching, rare classes rarely appear together, resulting in batches where $|P(i)| = 0$.
The **ClassBalancedSampler** (matching `FamilyBalancedSampler` from the reference paper) samples $P$ classes uniformly and $K$ instances per class:
$$\text{Batch Size} = P \times K$$
Every anchor is guaranteed exactly $K - 1$ positive instances in every mini-batch.

### 5.5 Proxy-Anchor Loss: Metric Learning via Class Proxies
Kim et al. (CVPR 2020) introduced **Proxy-Anchor Loss** to decouple metric learning from batch composition. Learnable class proxy vectors $\boldsymbol{P} \in \mathbb{R}^{C \times d}$ reside on $\mathbb{S}^{d-1}$:
$$\mathcal{L}_{\text{ProxyAnchor}} = \frac{1}{|P|} \sum_{p \in P} \log\left( 1 + \sum_{x \in X_p^+} e^{-\alpha (s(x, p) - \delta)} \right) + \frac{1}{|C|} \sum_{p \in C} \log\left( 1 + \sum_{x \in X_p^-} e^{\alpha (s(x, p) + \delta)} \right)$$
where:
- $\delta = 0.1$ is the margin.
- $\alpha = 32.0$ is the scaling factor.
- $s(x, p) = \boldsymbol{z}_x^\top \boldsymbol{p}$ is cosine similarity.
- $X_p^+$ is the set of positive data points for proxy $p$ in the batch; $X_p^-$ is the negative set.

**Theoretical Advantage:**
Proxies serve as global anchors for all 336 classes. Even if a batch has only one sample from a rare class, that sample is pulled directly toward its global proxy and repelled from all other class proxies, eliminating the sparse positive problem.

---

## 6. Sublinear Vector Serving in Continuous Metric Spaces

### 6.1 Hyperplane Locality-Sensitive Hashing (SimHash)
Draw $\ell = 128$ random Gaussian hyperplanes $\boldsymbol{G} \sim \mathcal{N}(0, \boldsymbol{I})$. Bit signatures are computed as $h(\boldsymbol{x}) = \operatorname{sign}(\boldsymbol{G} \boldsymbol{x})$.
$$\Pr[h(\boldsymbol{x}) = h(\boldsymbol{y})] = 1 - \frac{\theta(\boldsymbol{x}, \boldsymbol{y})}{\pi}$$

### 6.2 Hierarchical Navigable Small World (HNSW) Graphs
Constructs a multi-layer geometric skip-graph (Malkov & Yashunin, 2018):
- **$M = 32$:** Maximum bi-directional edges per node.
- **$efConstruction = 200$:** Candidate queue size during graph building.
- **$efSearch \in \{16, 50, 64, 100, 200\}$:** Dynamic candidate priority queue size during query routing. Greedy logarithmic descent achieves $\mathcal{O}(\log N)$ query time.

### 6.3 Two-Stage Retrieval: Shortlist Generation + Exact Reranking
1. **Stage 1 (ANN Shortlist):** Retrieve top-$M$ ($M=50$) candidates using fast HNSW in $\sim 0.15\text{ ms}$.
2. **Stage 2 (Exact Reranker):** Re-score the 50 candidates using exact inner product / Tanimoto similarity to produce the final top-$K$ ranking.

---

## 7. The Multi-Depth Factorial Ablation Matrix & Retrieval Metrics

### 7.1 Multi-Depth Evaluation Protocol
Directly matching `CS_328_Project_Report_previous.pdf`, we evaluate at multiple cutoff thresholds:
1. **Mean Average Precision at $k$ ($\text{MAP}@k$):**
   $$\text{AP}@k(q) = \frac{1}{\min(k, n_q)} \sum_{r=1}^k P_q(r) \cdot \operatorname{rel}_q(r), \quad \text{MAP}@k = \frac{1}{|Q|} \sum_{q \in Q} \text{AP}@k(q)$$
   evaluated for $k \in \{1, 5, 10, 50\}$.
2. **Recall at $k$ ($\text{Recall}@k$):**
   $$\text{Recall}@k(q) = \frac{|\operatorname{Top}_k(q) \cap R(q)|}{|R(q)|}$$
   evaluated for $k \in \{1, 5, 10, 50\}$.
3. **Mean Reciprocal Rank (MRR):**
   $$\text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\operatorname{rank}_1(q)}$$
4. **Retrieval-Derived Accuracy at 10 ($\text{Acc}@10$):**
   Majority vote prediction among the top-10 retrieved neighbors:
   $$\hat{y}_q = \operatorname{mode}\big( \{y_{c_1}, \dots, y_{c_{10}}\} \big), \quad \text{Acc}@10 = \frac{1}{|Q|} \sum_{q \in Q} \mathbb{I}(\hat{y}_q = y_q)$$

### 7.2 The Mathematical "Recall@10 Ceiling"
Let class size in the corpus be $n_q = |R(q)|$. Because $n_q$ averages $32.7$ (and up to $399$), retrieving only $k=10$ items imposes an inescapable mathematical upper bound:
$$\text{Recall}_{\max}@10 = \frac{1}{|Q|} \sum_{q \in Q} \frac{\min(10, n_q)}{n_q} = \mathbf{0.1696 \quad (17.0\%)}$$
Evaluating at $k=50$ removes this constraint ($\text{Recall}_{\max}@50 \approx 0.65$).

---
*End of Theoretical Foundations Document.*
