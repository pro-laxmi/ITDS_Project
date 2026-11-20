# Theoretical Foundations & Comprehensive Viva Guide: Scalable Molecular Graph Retrieval

**Course:** CS328: Introduction to Data Science, Semester V (Autumn 2026)  
**Institution:** Indian Institute of Technology Gandhinagar  
**Author:** Laxmidhar Panda (Roll No: 24110185)  
**Project:** Molecular Similarity Search Using Graph Neural Networks and Approximate Nearest Neighbors  
**Repository:** [https://github.com/pro-laxmi/ITDS_Project](https://github.com/pro-laxmi/ITDS_Project)  

---

## Executive Summary & Core Research Question

In early-stage computer-aided drug discovery and chemoinformatics, virtual screening relies on scanning libraries of millions to billions of candidate molecules to identify analogs sharing structural and biological characteristics with a query compound.

This project addresses the fundamental computational dilemma in molecular similarity retrieval:
1. **Combinatorial Exact Matching:** Testing whether two molecules share an exact structural motif is the **Subgraph Isomorphism** problem, which is proven **NP-complete**. Classical backtracking algorithms (e.g., Ullmann, VF2) exhibit exponential worst-case time complexity $\mathcal{O}(|V|^k)$, rendering linear exhaustive scans across large-scale chemical libraries computationally intractable.
2. **Classical Discrete Fingerprints:** Classical methods like Extended-Connectivity Fingerprints (**ECFP4** / circular Morgan fingerprints) hash radial topological neighborhoods into fixed-width binary bit vectors (e.g., 1024 bits) and score candidates using bitwise Tanimoto (Jaccard) similarity. While linear bitwise scans are fast, discrete fingerprints suffer from **rigid bit collisions**, **lossy bit folding**, and **topological inflexibility**—they cannot interpolate continuous chemical equivalence or learn metric embeddings tailored to structural scaffolds.
3. **The Geometric Deep Learning Hypothesis:** Can continuous graph representations learned via a **4-layer Graph Isomorphism Network (GIN)** mapped onto a 256-dimensional unit hypersphere $\mathbb{S}^{255}$ capture scaffold-level structural equivalence, outperforming classical discrete fingerprints under rigorous ground-truth evaluation while enabling sub-linear nearest-neighbor vector indexing?

---

## Table of Contents
1. [Molecular Graph Representation & Computational Complexity](#1-molecular-graph-representation--computational-complexity)
2. [Classical Discrete Chemoinformatics: ECFP4 Fingerprints & Tanimoto Metric Space](#2-classical-discrete-chemoinformatics-ecfp4-fingerprints--tanimoto-metric-space)
3. [Graph Neural Networks & Expressive Limits (The 1-WL Test)](#3-graph-neural-networks--expressive-limits-the-1-wl-test)
4. [GIN Encoder Architecture with Directional Edge Integration & Dual Readout](#4-gin-encoder-architecture-with-directional-edge-integration--dual-readout)
5. [Hyperspherical Geometry & Metric Isomorphism on $\mathbb{S}^{255}$](#5-hyperspherical-geometry--metric-isomorphism-on-mathbbs255)
6. [Training Label Taxonomy: Strategy B (Leak-Free Design)](#6-training-label-taxonomy-strategy-b-leak-free-design)
7. [Metric Learning Objectives & Mathematical Gradients](#7-metric-learning-objectives--mathematical-gradients)
   - 7.1 Cross-Entropy Classification (GIN-CE)
   - 7.2 Supervised Contrastive Learning (GIN-SupCon)
   - 7.3 GraphCL Stochastic View Augmentations (GIN-SupCon-Aug)
   - 7.4 Proxy-Anchor Metric Learning (GIN-ProxyAnchor)
   - 7.5 Mathematical Mechanics of Rare/Singleton Molecules: Negative-Repeller vs. Excluded
8. [Ground Truth Evaluation & The 27.5% Coverage Bound](#8-ground-truth-evaluation--the-275-coverage-bound)
9. [Locality-Sensitive Hashing & Vector Retrieval Backends](#9-locality-sensitive-hashing--vector-retrieval-backends)
10. [Disclosed Methodological Limitations](#10-disclosed-methodological-limitations)
11. [The "Why" of Everything: Design Decisions Explained](#11-the-why-of-everything-design-decisions-explained)
12. [Comprehensive Viva Defense & Oral Exam Question Bank (25 Questions & Answers)](#12-comprehensive-viva-defense--oral-exam-question-bank-25-questions--answers)

---

## 1. Molecular Graph Representation & Computational Complexity

### 1.1 Molecules as Attributed Graphs
A molecule is naturally represented as an undirected attributed graph:
$$\mathcal{G} = (\mathcal{V}, \mathcal{E}, \boldsymbol{X}_v, \boldsymbol{E}_{uv})$$
where:
* $\mathcal{V} = \{v_1, v_2, \dots, v_{|\mathcal{V}|}\}$ is the set of vertices representing atoms.
* $\mathcal{E} \subseteq \mathcal{V} \times \mathcal{V}$ is the set of edges representing covalent chemical bonds.
* $\boldsymbol{X}_v$ represents categorical node features: discrete atomic number tokens $x_v \in [0, 27]$ (encoding 28 atom species: C, N, O, S, F, P, Cl, Br, I, etc., with formal charge and protonation states) plus chiral tags.
* $\boldsymbol{E}_{uv}$ represents discrete edge features: bond type $e_{uv} \in \{1, 2, 3\}$ denoting single, double, and triple covalent bonds, embedded through categorical embedding matrices.

### 1.2 Exact Subgraph Isomorphism & NP-Completeness
Given a query molecular graph $\mathcal{G}_q = (\mathcal{V}_q, \mathcal{E}_q)$ and a database target graph $\mathcal{G}_t = (\mathcal{V}_t, \mathcal{E}_t)$, the **subgraph isomorphism problem** seeks an injective mapping $f: \mathcal{V}_q \to \mathcal{V}_t$ such that:
1. $\forall u \in \mathcal{V}_q, \boldsymbol{X}_q(u) = \boldsymbol{X}_t(f(u))$ (matching node attributes).
2. $\forall (u, v) \in \mathcal{E}_q, (f(u), f(v)) \in \mathcal{E}_t$ and $\boldsymbol{E}_q(u, v) = \boldsymbol{E}_t(f(u), f(v))$ (matching edge attributes).

**Computational Complexity:**
* Subgraph isomorphism is proven **NP-complete** (Ullmann, 1976; Cordella et al., 2004). *(Note: Cook, 1971 proved NP-completeness of SAT; Subgraph Isomorphism NP-completeness is established by Karp, 1972 and Ullmann, 1976).*
* Classical state-space backtracking algorithms (Ullmann, 1976; VF2, Cordella et al., 2004) exhibit exponential worst-case time complexity:
  $$\mathcal{O}\left(|\mathcal{V}_t|^{|\mathcal{V}_q|} \cdot |\mathcal{V}_q|\right)$$
* Even with chemical heuristic pruning (e.g., screening by atom counts and bond types), checking a query against $N = 10^7$ molecules sequentially would take hours or days per query, making real-time interactive retrieval impossible.

### 1.3 Graph Edit Distance (GED)
When looking for *similar* rather than exact subgraphs, the standard mathematical formulation is the **Graph Edit Distance (GED)**:
$$\operatorname{GED}(\mathcal{G}_1, \mathcal{G}_2) = \min_{(e_1, \dots, e_k) \in \Upsilon(\mathcal{G}_1, \mathcal{G}_2)} \sum_{i=1}^k c(e_i)$$
where $\Upsilon(\mathcal{G}_1, \mathcal{G}_2)$ is the set of all edit paths transforming $\mathcal{G}_1$ into $\mathcal{G}_2$ using basic operations (vertex/edge insertion, deletion, and substitution), with non-negative cost function $c(e_i)$.
* Computing exact GED is **NP-hard**, and approximating it within a constant factor remains APX-hard.

---

## 2. Classical Discrete Chemoinformatics: ECFP4 Fingerprints & Tanimoto Metric Space

### 2.1 The Extended-Connectivity Fingerprint (ECFP / Morgan Algorithm)
The industrial gold standard in chemoinformatics is the **Extended-Connectivity Fingerprint (ECFP)** (Rogers & Hahn, 2010), rooted in the Morgan algorithm (1965). ECFP4 corresponds to diameter $d = 4$, which is radius $r = 2$ circular atom neighborhoods.

### 2.2 Algorithm Stages
1. **Stage 0 (Atom Invariants Initialization):** Each atom $v$ is assigned an initial 32-bit integer identifier based on its chemical properties:
   $$h_v^{(0)} = \operatorname{Hash}\Big(\text{AtomicNumber}(v), \text{HeavyNeighbors}(v), \text{Valence}(v), \text{FormalCharge}(v), \text{Aromaticity}(v)\Big)$$
2. **Stage 1 to $r$ (Radial Iterative Expansion):** For each radius step $t = 1, \dots, r$:
   - Collect neighborhood tuples:
     $$T_v^{(t)} = \Big\{ \big(\text{BondOrder}(u, v), h_u^{(t-1)}\big) \;\Big|\; u \in \mathcal{N}(v) \Big\}$$
   - Sort tuples deterministically to ensure graph isomorphism invariance:
     $$h_v^{(t)} = \operatorname{Hash}\Big( h_v^{(t-1)}, \operatorname{sort}\big(T_v^{(t)}\big) \Big)$$
3. **Stage 3 (Shingle Set Collection & Duplicate Removal):**
   $$\mathcal{S}(\mathcal{G}) = \bigcup_{t=0}^r \bigcup_{v \in \mathcal{V}} \big\{ h_v^{(t)} \big\}$$
   Each unique 32-bit integer in $\mathcal{S}(\mathcal{G})$ represents a specific circular topological shingle.

### 2.3 Folding into Fixed-Width Binary Vectors
To map the unbounded sparse set $\mathcal{S}(\mathcal{G})$ into a dense fixed-width binary vector $\boldsymbol{x} \in \{0, 1\}^D$ (typically $D = 1024$ bits):
$$x_j = \mathbb{I}\Big(\exists s \in \mathcal{S}(\mathcal{G}) \text{ s.t. } (s \bmod D) = j\Big)$$

### 2.4 Tanimoto (Jaccard) Similarity Metric Space
For two binary fingerprint vectors $\boldsymbol{u}, \boldsymbol{v} \in \{0, 1\}^D$, the **Tanimoto coefficient** is:
$$T(\boldsymbol{u}, \boldsymbol{v}) = \frac{\boldsymbol{u} \cdot \boldsymbol{v}}{\|\boldsymbol{u}\|_1 + \|\boldsymbol{v}\|_1 - \boldsymbol{u} \cdot \boldsymbol{v}} = \frac{c}{a + b - c}$$
where:
* $a = \|\boldsymbol{u}\|_1$ is the number of bits set in $\boldsymbol{u}$.
* $b = \|\boldsymbol{v}\|_1$ is the number of bits set in $\boldsymbol{v}$.
* $c = \boldsymbol{u} \cdot \boldsymbol{v}$ is the number of intersecting common bits ($a \cap b$).

**Metric Property:** The distance $d_T(\boldsymbol{u}, \boldsymbol{v}) = 1 - T(\boldsymbol{u}, \boldsymbol{v})$ satisfies the triangle inequality and is a formal metric on $\{0, 1\}^D$ (Levandowsky & Winter, 1971).

### 2.5 Inherent Limitations of Classical Fingerprints
1. **Rigid Bit Collisions:** Distinct chemical subgraphs that hash to the same value modulo $D$ collide, producing false-positive similarity signals.
2. **Topological Inflexibility:** ECFP considers bit positions as orthogonal and independent. A single minor substitution (e.g., swapping a benzene ring for a pyridine ring) alters multiple circular neighborhood hashes across radii 0, 1, and 2, causing an abrupt drop in Tanimoto similarity despite near-identical biological activity.
3. **Absence of Learning:** Classical fingerprints cannot adapt their representation to the task or learn continuous embeddings organized by structural scaffolds.

---

## 3. Graph Neural Networks & Expressive Limits (The 1-WL Test)

### 3.1 The Weisfeiler-Lehman (1-WL) Graph Isomorphism Test
The 1-WL test is a classical combinatorial heuristic for graph isomorphism:
1. Initialize each node with a color $c_v^{(0)} = \operatorname{hash}(\text{label}(v))$.
2. At each iteration $l$, update node color by aggregating the multiset of neighbor colors:
   $$c_v^{(l)} = \operatorname{hash}\left( c_v^{(l-1)}, \{\!\{ c_u^{(l-1)} \mid u \in \mathcal{N}(v) \}\!\} \right)$$
3. Count color histograms across the graph. If two graphs have different color histograms at any iteration, they are guaranteed to be non-isomorphic. If their histograms match, they are potentially isomorphic (1-WL is a necessary, but not always sufficient, condition).

### 3.2 Expressive Limits of Standard MPNNs (Xu et al., ICLR 2019)
Standard Graph Convolutional Networks (GCN) and GraphSAGE use **mean** or **max** pooling for neighborhood aggregation:
* **Mean Pooling Failure:** $\operatorname{mean}(\{\!\{a, a\}\!\}) = \operatorname{mean}(\{\!\{a, a, a\}\!\}) = a$. Mean aggregation captures the proportion of neighbor types but completely loses graph scale and neighbor count/multiplicity.
* **Max Pooling Failure:** $\operatorname{max}(\{\!\{a, b\}\!\}) = \operatorname{max}(\{\!\{a, b, b\}\!\}) = \max(a, b)$. Max pooling identifies distinct feature types but ignores frequency entirely.

**Theorem (Xu et al., ICLR 2019):**  
*A Message Passing Neural Network has at most the discriminative power of the 1-WL test. It achieves maximal 1-WL expressive power if and only if its multiset aggregation function and graph-level readout function are injective.*

### 3.3 The Graph Isomorphism Network (GIN) Formulation
To guarantee injectivity over countable multisets, Xu et al. proved that sum-based multiset aggregation with a Multi-Layer Perceptron (MLP) universal approximator satisfies the condition:
$$h_v^{(l)} = \operatorname{MLP}^{(l)}\left( (1 + \epsilon^{(l)}) h_v^{(l-1)} + \sum_{u \in \mathcal{N}(v)} h_u^{(l-1)} \right)$$
where $\epsilon^{(l)}$ is a learnable or fixed scalar.

---

## 4. GIN Encoder Architecture with Directional Edge Integration & Dual Readout

### 4.1 Directional Bond-Featured Message Passing (GINEConv)
In molecular chemistry, bond types (single, double, triple) are fundamental to structural connectivity. We implement directional edge-featured message passing:
$$m_{uv}^{(l)} = \operatorname{ReLU}\left( h_u^{(l-1)} + \boldsymbol{W}_e^{(l)} \boldsymbol{e}_{uv} \right)$$
$$h_v^{(l)} = \operatorname{MLP}^{(l)}\left( (1 + \epsilon^{(l)}) h_v^{(l-1)} + \sum_{u \in \mathcal{N}(v)} m_{uv}^{(l)} \right)$$
where:
* $\boldsymbol{e}_{uv} \in \mathbb{R}^{64}$ is the categorical bond embedding vector.
* $\boldsymbol{W}_e^{(l)} \in \mathbb{R}^{64 \times 64}$ is a linear edge-projection matrix matching the node hidden dimension.
* The message combining sender node $h_u^{(l-1)}$ and bond feature $\boldsymbol{e}_{uv}$ is passed through a **$\operatorname{ReLU}$** activation before multiset summation.
* $\operatorname{MLP}^{(l)}$ is a 2-layer perceptron (expanding to 128 hidden dimensions):
  $$\operatorname{Linear}(64 \to 128) \to \operatorname{BatchNorm1d}(128) \to \operatorname{GELU} \to \operatorname{Linear}(128 \to 64) \to \operatorname{BatchNorm1d}(64)$$
* A residual skip connection with Dropout ($p = 0.2$) is applied after each layer:
  $$h_v^{(l)} \leftarrow \operatorname{ReLU}(h_v^{(l-1)} + h_v^{(l)})$$

### 4.2 Dual Global Graph Readout Pooling
A molecular embedding must capture both:
1. **Extensive Properties (Graph Size & Atom Counts):** Captured by multiset sum pooling:
   $$\boldsymbol{h}_{\text{sum}} = \sum_{v \in \mathcal{V}} h_v^{(4)} \in \mathbb{R}^{64}$$
2. **Intensive Properties (Structural Density & Relative Topology):** Captured by mean pooling:
   $$\boldsymbol{h}_{\text{mean}} = \frac{1}{|\mathcal{V}|} \sum_{v \in \mathcal{V}} h_v^{(4)} \in \mathbb{R}^{64}$$

To preserve both properties simultaneously without loss of information, we concatenate both pooling operations:
$$\boldsymbol{h}_G = \big[ \boldsymbol{h}_{\text{sum}} \;\big\|\; \boldsymbol{h}_{\text{mean}} \big] \in \mathbb{R}^{128}$$

### 4.3 Projection Head
The concatenated graph representation $\boldsymbol{h}_G \in \mathbb{R}^{128}$ is mapped into the final metric retrieval space via a 2-layer projection head with LayerNorm, GELU, and Dropout:
$$\tilde{\boldsymbol{z}}_G = \operatorname{Linear}_{256 \to 256}\Big(\operatorname{Dropout}_{0.2}\big(\operatorname{GELU}\big(\operatorname{LayerNorm}_{256}(\operatorname{Linear}_{128 \to 256}(\boldsymbol{h}_G))\big)\big)\Big) \in \mathbb{R}^{256}$$

---

## 5. Hyperspherical Geometry & Metric Isomorphism on $\mathbb{S}^{255}$

### 5.1 $L_2$ Unit Hyperspherical Normalization
The unconstrained projection vector $\tilde{\boldsymbol{z}}_G$ is projected onto the unit hypersphere $\mathbb{S}^{255}$:
$$\boldsymbol{z} = \frac{\tilde{\boldsymbol{z}}_G}{\|\tilde{\boldsymbol{z}}_G\|_2} \in \mathbb{S}^{255}, \quad \|\boldsymbol{z}\|_2 = 1$$

### 5.2 The Euclidean-Cosine Isomorphism Proof
Constraining representations to $\mathbb{S}^{255}$ provides a rigorous mathematical bridge between cosine similarity and Euclidean nearest-neighbor search:

**Theorem:** For any two unit-normalized vectors $\boldsymbol{q}, \boldsymbol{c} \in \mathbb{S}^{255}$ ($\|\boldsymbol{q}\|_2 = \|\boldsymbol{c}\|_2 = 1$):
$$\|\boldsymbol{q} - \boldsymbol{c}\|_2^2 = 2 - 2 \langle \boldsymbol{q}, \boldsymbol{c} \rangle$$
**Proof:**
$$\|\boldsymbol{q} - \boldsymbol{c}\|_2^2 = \langle \boldsymbol{q} - \boldsymbol{c}, \boldsymbol{q} - \boldsymbol{c} \rangle = \|\boldsymbol{q}\|_2^2 + \|\boldsymbol{c}\|_2^2 - 2 \langle \boldsymbol{q}, \boldsymbol{c} \rangle$$
Since $\|\boldsymbol{q}\|_2 = 1$ and $\|\boldsymbol{c}\|_2 = 1$:
$$\|\boldsymbol{q} - \boldsymbol{c}\|_2^2 = 1 + 1 - 2 \langle \boldsymbol{q}, \boldsymbol{c} \rangle = 2 - 2 \langle \boldsymbol{q}, \boldsymbol{c} \rangle \quad \blacksquare$$

**Implications for Vector Retrieval:**
1. **Strict Monotonicity:** Maximizing inner product $\langle \boldsymbol{q}, \boldsymbol{c} \rangle$ is strictly equivalent to minimizing Euclidean distance $\|\boldsymbol{q} - \boldsymbol{c}\|_2$.
2. **Zero Distortion in Indexing:** Fast vector search libraries (such as FAISS `IndexFlatIP`) can rank candidates by simple dot product, guaranteeing mathematically identical nearest-neighbor rankings to Euclidean metric indices with zero geometric distortion.
3. **Elimination of Magnitude Bias:** Molecules with large atom counts cannot dominate retrieval rankings simply due to larger embedding norm vectors; similarity is determined purely by angular orientation on $\mathbb{S}^{255}$.

---

## 6. Training Label Taxonomy: Strategy B (Leak-Free Design)

### 6.1 The Data Leakage Pitfall (Strategy A)
In metric learning, the training label taxonomy defines which pairs of molecules are pulled together. A severe pitfall in prior work (Strategy A) was **training-label leakage**:
* Canonical Murcko scaffolds were extracted across the *entire 12,000-molecule dataset* (including validation and test sets) to establish frequency cutoffs and class IDs.
* Consequently, scaffold frequencies and class assignments in the training set were influenced by test set distributions.
* Furthermore, test molecules that appeared only once or twice in the test set were artificially grouped based on global statistics.

### 6.2 Construction of Strategy B (Strictly Leak-Free)
To ensure absolute scientific integrity, **Strategy B** is constructed strictly from the **10,000 training molecules**:
1. **Extraction:** Canonical Bemis-Murcko scaffolds were extracted from the 10,000 training molecules using RDKit.
2. **Frequent Scaffolds ($\ge 3$ occurrences):** Only scaffolds appearing $\ge 3$ times in the training split were assigned supervised class labels. This yielded exactly **241 genuine non-empty Murcko scaffold classes** (Classes $0 - 240$), encompassing **1,839 molecules**.
3. **Auxiliary Acyclic Pseudo-Classes:** Exactly 55 training molecules are acyclic (possessing an empty Murcko scaffold `""`). Rather than dropping them or creating a single massive, chemically heterogeneous class of all linear molecules, they were partitioned into **3 auxiliary acyclic pseudo-classes** (Classes $241, 242, 243$) using deterministic round-robin assignment:
   $$y_i = 241 + (i \bmod 3)$$
   This provides supervised positive pairs among acyclic molecules while preventing false positive associations between unrelated linear structures and cyclic scaffolds.
4. **Unclustered Rare/Singleton Scaffolds (Frequency $< 3$):** Scaffolds appearing only 1 or 2 times cannot form reliable positive contrastive pairs (or would form trivial singletons). These **8,106 molecules** were assigned class label **-1**.
5. **Total Supervised Corpus:** Exactly **244 classes** (241 genuine Murcko + 3 acyclic pseudo-classes) supported by **1,894 supervised molecules**.

| Category | Class Label IDs | Distinct Scaffolds | Molecules | Role in Supervised Training |
| :--- | :---: | :---: | :---: | :--- |
| **Frequent Scaffolds ($\ge 3$)** | $0 - 240$ | 241 | 1,839 | Form supervised positive pairs on unit sphere |
| **Acyclic Pseudo-Classes** | $241 - 243$ | 1 (Empty `""`) | 55 | Form auxiliary positive pairs for linear molecules |
| **Unclustered Rare/Singleton** | -1 | 4,263 | 8,106 | Excluded from positive loss; optional negative repellers |
| **Total Training Split** | **244 Classes + {-1}** | **4,505** | **10,000** | Complete, leak-free training split |

---

## 7. Metric Learning Objectives & Mathematical Gradients

All models were trained for a fixed 20 epochs using the Adam optimizer ($\text{lr} = 10^{-3}$, weight decay $= 10^{-5}$) with a batch size of 128.

### 7.1 Cross-Entropy Classification (GIN-CE)
A linear classification layer $\boldsymbol{W}_{\text{cls}} \in \mathbb{R}^{244 \times 256}$ is appended to the 256-D normalized embeddings:
$$\mathcal{L}_{\text{CE}} = -\frac{1}{|B_{\text{sup}}|} \sum_{i \in B_{\text{sup}}} \log \frac{\exp(\boldsymbol{w}_{y_i}^\top \boldsymbol{z}_i)}{\sum_{k=0}^{243} \exp(\boldsymbol{w}_k^\top \boldsymbol{z}_i)}$$
where $B_{\text{sup}}$ consists only of samples with label $y_i \in [0, 243]$ (label -1 excluded).

**Why GIN-CE Failed Retrieval (Covered MAP@10 = 0.2423):**  
Cross-entropy optimizes for linear hyperplane separability. Once an embedding is on the correct side of the decision boundary, the loss gradient approaches zero. Cross-entropy does not enforce that molecules within the same scaffold class cluster tightly together on the hypersphere, leading to diffuse intra-class clusters and poor nearest-neighbor retrieval.

### 7.2 Supervised Contrastive Learning (GIN-SupCon)
Adapts Khosla et al. (NeurIPS 2020) to molecular graphs using class-balanced batch sampling:
$$\mathcal{L}_{\text{SupCon}} = \sum_{i \in B_{\text{sup}}} \frac{-1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(\langle \boldsymbol{z}_i, \boldsymbol{z}_p \rangle / \tau)}{\sum_{a \in A(i)} \exp(\langle \boldsymbol{z}_i, \boldsymbol{z}_a \rangle / \tau)}$$
where:
* $P(i) = \{p \in A(i) \mid y_p = y_i, p \neq i\}$ is the set of positive samples sharing class $y_i$ with anchor $i$.
* $A(i)$ is the contrastive cohort (all other samples in the mini-batch).
* Temperature $\tau = 0.07$ acts as a hardness parameter.

**Gradient Dynamics & Why GIN-SupCon Won (Covered MAP@10 = 0.5996):**  
The gradient with respect to anchor $\boldsymbol{z}_i$ is:
$$\nabla_{\boldsymbol{z}_i} \mathcal{L}_{\text{SupCon}} = \frac{1}{\tau} \sum_{p \in P(i)} \left( \sum_{a \in A(i)} P_{ia} \boldsymbol{z}_a - \boldsymbol{z}_p \right)$$
where $P_{ia} = \frac{\exp(\langle \boldsymbol{z}_i, \boldsymbol{z}_a \rangle / \tau)}{\sum_{b} \exp(\langle \boldsymbol{z}_i, \boldsymbol{z}_b \rangle / \tau)}$ is the softmax probability.
* For negative impostors with high cosine similarity ($\langle \boldsymbol{z}_i, \boldsymbol{z}_a \rangle \approx 1$), $P_{ia}$ is large, generating massive repulsive gradient forces.
* For positive pairs, the term $-\boldsymbol{z}_p$ pulls them directly together.
* Small $\tau = 0.07$ heavily penalizes hard negatives, carving out tight, isolated hyperspherical clusters for each of the 241 Murcko scaffolds.

### 7.3 GraphCL Stochastic View Augmentations (GIN-SupCon-Aug)
Applies stochastic graph augmentations (You et al., NeurIPS 2020) during SupCon training:
* **Edge Dropping:** Covalent bonds randomly dropped with probability $p = 0.1$.
* **Feature Masking:** Atom tokens randomly masked with probability $p = 0.1$.

**Why Augmentation Severely Degraded Performance (Covered MAP@10 = 0.2928, -51.2% Drop):**  
In natural image contrastive learning (SimCLR), cropping or color jitter preserves semantic identity (a dog is still a dog).  
In chemistry, **covalent bonds define the core Murcko scaffold**:
* Dropping an aromatic bond in a chlorobenzene ring cleaves the cycle, turning a rigid aromatic core into an open acyclic chain.
* Deleting a heteroatom changes hydrogen bonding and ring aromaticity.
* Consequently, stochastic augmentations actively destroy the canonical Bemis-Murcko scaffold, generating augmented views that belong to a *different scaffold identity* than the anchor while forcing the model to pull them together. This injects contradictory supervisory noise directly into the contrastive objective.

### 7.4 Proxy-Anchor Metric Learning (GIN-ProxyAnchor)
Introduced by Kim et al. (CVPR 2020), Proxy-Anchor loss associates a learnable proxy vector $\boldsymbol{p}_k \in \mathbb{S}^{255}$ with each class $k \in \{0, \dots, 243\}$. Proxies reside on the unit hypersphere and act as persistent global class centroids:
$$\mathcal{L}_{\text{PA}} = \frac{1}{|P|} \sum_{p \in P} \log\left( 1 + \sum_{x \in X_p^+} e^{-\alpha (\langle \boldsymbol{z}_x, \boldsymbol{p} \rangle - \delta)} \right) + \frac{1}{|C|} \sum_{p \in C} \log\left( 1 + \sum_{x \in X_p^-} e^{\alpha (\langle \boldsymbol{z}_x, \boldsymbol{p} \rangle + \delta)} \right)$$
where:
* $\alpha = 32.0$ is the scaling factor.
* $\delta = 0.10$ is the metric margin.
* $C$ is the set of all 244 class proxies; $P \subseteq C$ is the subset of proxies with at least one positive sample in the mini-batch ($X_p^+ \neq \emptyset$).
* $X_p^+$ is the set of positive data points for proxy $p$; $X_p^-$ is the set of negative data points for proxy $p$.

### 7.5 Mathematical Mechanics of Rare Molecules (Label -1): Negative-Repeller vs. Excluded
In Strategy B, 8,106 training molecules have label -1 (rare/singleton scaffolds). We evaluated two distinct mathematical formulations:

#### 1. ProxyAnchor Excluded (`GIN-ProxyAnchor (Excluded)`)
Molecules with label -1 are completely omitted from the batch during loss calculation:
* Only the 1,894 labeled molecules contribute to proxy updates.
* **Result: Covered MAP@10 = 0.5593, Recall@10 = 0.5244.**

#### 2. ProxyAnchor Negative-Repeller (`GIN-ProxyAnchor (Neg-Repeller)`)
Molecules with label -1 have no positive class proxy ($x \notin X_p^+$ for all $p$). However, they are treated as negative instances for *all 244 class proxies* ($x \in X_p^-$ for all $p \in C$):
* **Gradient on Proxy $\boldsymbol{p}$ from Negative Repellers:**
  $$\frac{\partial \mathcal{L}_{\text{neg}}}{\partial \boldsymbol{p}} = \frac{1}{|C|} \frac{\alpha \sum_{x \in X_p^-} e^{\alpha (\langle \boldsymbol{z}_x, \boldsymbol{p} \rangle + \delta)} \boldsymbol{z}_x}{1 + \sum_{x \in X_p^-} e^{\alpha (\langle \boldsymbol{z}_x, \boldsymbol{p} \rangle + \delta)}}$$
* This pushes proxy $\boldsymbol{p}$ away from the embedding $\boldsymbol{z}_x$ of the rare molecule.
* **Result: Covered MAP@10 = 0.5418, Recall@10 = 0.5127.**

**Why Excluded Outperformed Negative-Repeller (+0.0175 MAP@10):**  
Forcing 8,106 structurally diverse rare molecules to repel all 244 class proxies simultaneously introduced diffuse, uncoordinated gradient repulsion across the entire hypersphere. This background noise compressed the angular margins between genuine class proxies, slightly degrading retrieval fidelity compared to completely excluding unclustered samples.

---

## 8. Ground Truth Evaluation & The 27.5% Coverage Bound

### 8.1 Bemis-Murcko Ground Truth Relevance
Ground-truth relevance is evaluated against canonical Bemis-Murcko (BM) scaffolds extracted via RDKit (`MurckoScaffold.GetScaffoldForMol`):
* A retrieved candidate molecule $c$ is relevant to query $q$ if and only if both share the exact same non-empty canonical BM scaffold key:
  $$\operatorname{Rel}(q, c) = \begin{cases} 1 & \text{if } \operatorname{BM}(q) = \operatorname{BM}(c) \text{ and } \operatorname{BM}(q) \neq \text{""} \\ 0 & \text{otherwise} \end{cases}$$
* Acyclic molecules (empty scaffold `""`) receive relevance 0 to prevent artificial true positives between unrelated linear chains.

### 8.2 The Query Coverage Partition
The candidate retrieval corpus $\mathcal{C}$ consists of 11,000 molecules (10,000 train + 1,000 validation).  
The evaluation query set $\mathcal{Q}$ consists of 1,000 independent test molecules.

A critical property of the ZINC dataset is that molecular scaffolds follow a heavy-tailed power-law distribution. Consequently:
* **Covered Test Queries ($N = 275$):** Exactly 275 test queries possess at least one molecule in the 11,000-candidate corpus sharing their exact canonical Murcko scaffold ($|C_q^*| \ge 1$).
* **Uncovered Test Queries ($N = 725$):** Exactly 725 test queries possess unique Murcko scaffolds that appear **zero times** anywhere in the 11,000-molecule corpus ($|C_q^*| = 0$).

### 8.3 Mathematical Derivation of the Full-Test Upper Bound Ceiling
For an uncovered query ($|C_q^*| = 0$), there are zero true positives in the candidate pool. Therefore, **no retrieval system, regardless of architecture or capacity, can achieve an AP@10 or Recall@10 greater than 0.0**. Under standard evaluation protocol, these 725 queries receive a score of exactly 0.0.

The theoretical maximum possible Mean Average Precision across the entire 1,000-query test set is:
$$\max \operatorname{MAP}_{\text{full}} = \frac{1}{|\mathcal{Q}|} \sum_{q \in \mathcal{Q}} \max \operatorname{AP}@10(q) = \frac{1}{1000} \left( \sum_{q \in \mathcal{Q}_{\text{cov}}} 1.0 + \sum_{q \in \mathcal{Q}_{\text{uncov}}} 0.0 \right)$$
$$\max \operatorname{MAP}_{\text{full}} = \frac{275 \times 1.0 + 725 \times 0.0}{1000} = \mathbf{0.2750 \quad (27.5\%)}$$

Similarly, the maximum possible Recall@10 across the full test set is strictly bounded by $\mathbf{0.2750}$.

**Why Disaggregating Covered ($N=275$) vs. Full Test ($N=1000$) is Essential:**  
Reporting only the full-test MAP@10 (0.1649 for GIN-SupCon vs. 0.0631 for ECFP4) obscures model retrieval quality due to $72.5\%$ zero-inflation from queries where no match exists. Evaluating the 275 covered queries isolates algorithmic ranking fidelity:
* GIN-SupCon achieves **0.5996 MAP@10** on covered queries vs. **0.2294 for ECFP4** (+161.4% relative gain).

---

## 9. Locality-Sensitive Hashing & Vector Retrieval Backends

### 9.1 Classical Bitwise Baseline: Exhaustive Linear Tanimoto Scan
* Generates 1024-bit Morgan fingerprints using RDKit (`GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)`).
* Searches the 11,000-molecule corpus using an exhaustive pairwise bitwise popcount Tanimoto scan.
* Serving latency: $\approx 1.25\text{ ms}$ per query on CPU.

### 9.2 Learned Vector Search: FAISS IndexFlatIP (Exact Inner Product)
* 11,000 corpus molecules are passed through the trained GIN encoder; the resulting 256-D hyperspherical embeddings are indexed in FAISS using `IndexFlatIP`.
* Because all vectors are unit-normalized ($\|\boldsymbol{z}\|_2 = 1$), FAISS `IndexFlatIP` computes exact inner products $\langle \boldsymbol{q}, \boldsymbol{c} \rangle$, which is mathematically isomorphic to exact Euclidean nearest-neighbor search.
* Serving latency: $\approx 0.89\text{ ms}$ per query on CPU.

### 9.3 Exploratory Indexing: FAISS IndexHNSWFlat
* Proximity-graph approximate nearest neighbor index with parameters $M = 32$, $efConstruction = 200$, $efSearch = 64$.
* Disclosed in the report as an **exploratory implementation**. On this 11,000-molecule corpus, linear scans and flat inner product searches are already sub-millisecond, so HNSW is retained for future scalability to multi-million molecule libraries.

---

## 10. Disclosed Methodological Limitations

In strict adherence to scientific rigor, the following 9 methodological boundaries are disclosed:
1. **Single-Run Benchmark Results:** All reported metrics originate from evaluations on fixed saved checkpoints. While verified against serialized JSON artifacts, multi-seed statistical significance is not claimed.
2. **Fixed Epoch Training (No Validation Checkpoint Selection):** Strategy B models were trained for a fixed 20 epochs; checkpoint selection via lowest validation retrieval loss was not implemented in the training script.
3. **Scaffold Key Artifact Provenance:** Ground-truth relevance relies on precomputed keys stored in `data/exact_scaffolds.pkl`. A self-contained script to re-extract these keys from scratch is omitted from the repository.
4. **Multi-Seed Script Objective Discrepancy:** The multi-seed validation script (`src/multiseed_strategy_b.py`) trains the Neg-Repeller objective, not the Excluded objective, and thus does not measure Excluded model variance.
5. **Chemical Sanitization Protocol:** Ingestion pipelines rely on PyTorch Geometric ZINC conventions; secondary RDKit valence sanitization (`SanitizeMol`) was not explicitly applied across all raw inputs.
6. **Exploratory Status of HNSW Indexing:** HNSW is implemented in the codebase as an exploratory feature; no empirical speedup claims over FlatIP are made.
7. **2D Scaffold Matching vs. 3D Bioactivity:** Bemis-Murcko scaffold matching evaluates 2D topological connectivity; it does not model 3D conformational energetics or true biological binding affinity.
8. **Benchmark Scale:** Experiments evaluate a curated 12,000-molecule ZINC subset; scaling behavior on multi-million or billion-scale libraries remains unproven.
9. **Causal Attributions:** Hypotheses regarding graph augmentation degradation and proxy repulsion mechanics are interpretations of empirical differences, not independently proven causal mechanisms.

---

## 11. The "Why" of Everything: Design Decisions Explained

| Decision | What We Did | Why We Did It | Coherent Evidence / Result |
| :--- | :--- | :--- | :--- |
| **Why GIN over GCN/GAT?** | Implemented 4-layer GINEConv with sum pooling. | GCN/GAT use mean/attention pooling, which fails the 1-WL multiset injectivity test. Sum pooling can distinguish molecules with identical atom proportions but different sizes. | GIN achieves maximal 1-WL expressiveness over graph multisets (Xu et al., 2019). |
| **Why Directional Bond Features?** | Integrated covalent bond embeddings into message function via ReLU. | Chemical bond orders (single, double, triple) fundamentally define molecular topology and reactivity. | Preserves directional bond features directly in edge-aware message passing. |
| **Why Dual Global Readout?** | Concatenated `global_add_pool` (64-D) and `global_mean_pool` (64-D). | Sum pooling captures molecular size (atom count); mean pooling captures atom density and relative structural composition. | Combines extensive and intensive geometric invariants into a unified 128-D graph representation. |
| **Why Hyperspherical Normalization ($\mathbb{S}^{255}$)?** | Constrained all embeddings to $\|\boldsymbol{z}\|_2 = 1$. | Provides exact mathematical isomorphism: $\|\boldsymbol{q}-\boldsymbol{c}\|_2^2 = 2 - 2\langle \boldsymbol{q}, \boldsymbol{c} \rangle$. Eliminates vector norm bias and allows FAISS `IndexFlatIP` to perform exact Euclidean nearest neighbor retrieval. | Proved analytically in Section 5.2. Enables sub-millisecond retrieval. |
| **Why Switch from Strategy A to Strategy B?** | Abandoned dataset-wide scaffold clustering; restricted class creation strictly to the 10,000 training molecules. | Strategy A suffered from severe data leakage: test set molecules influenced scaffold frequency thresholds and class IDs. Strategy B eliminates all leakage. | 100% leak-free training label taxonomy. |
| **Why the $\ge 3$ Frequency Threshold?** | Required scaffolds to appear $\ge 3$ times in train to become a class. | Contrastive learning requires positive pairs. Scaffolds with 1 or 2 molecules cannot form diverse positive contrastive batches. | Yields 241 genuine Murcko scaffold classes (1,839 molecules). |
| **Why 3 Acyclic Pseudo-Classes?** | Partitioned 55 acyclic molecules into 3 groups via round-robin $(y_i = 241 + (i \bmod 3))$. | 55 molecules had empty scaffold `""`. Grouping all linear molecules into one class would falsely pull unrelated linear chains together; dropping them would discard valid molecules. | Disclosed as an auxiliary training-label design choice; excluded from ground-truth scaffold evaluation. |
| **Why Exclude 8,106 Rare Molecules from SupCon?** | Assigned label -1 to rare/singleton scaffolds. | Rare molecules cannot form valid intra-class positive contrastive pairs; forcing them into singleton classes causes batch collapse. | Avoids contradictory positive pair supervision. |
| **Why Experiment with ProxyAnchor?** | Implemented learnable class proxy vectors $\boldsymbol{p}_k \in \mathbb{S}^{255}$. | In long-tailed molecular distributions, batch-dependent contrastive learning suffers when positive pairs are absent from a mini-batch. Proxies act as static global anchors. | ProxyAnchor achieved second-highest performance (0.5593 MAP@10), outperforming CE (0.2423) and ECFP4 (0.2294). |
| **Why Did SupCon Win over ProxyAnchor?** | SupCon achieved 0.5996 MAP@10 vs. 0.5593 for ProxyAnchor. | SupCon directly optimizes pairwise sample-to-sample relationships in class-balanced mini-batches with hard-negative penalization ($\tau = 0.07$), whereas ProxyAnchor optimizes sample-to-proxy distances, introducing slight geometric softening. | Measured empirical difference: $+7.2\%$ relative gain of SupCon over ProxyAnchor. |
| **Why Did GraphCL Augmentations Fail?** | SupCon-Aug dropped from 0.5996 to 0.2928 MAP@10. | Randomly deleting covalent bonds or masking atoms shatters cyclic rings, transforming the canonical Murcko scaffold into an entirely different chemical identity. | Injects contradictory label noise into contrastive pairs. |
| **Why Disaggregate Covered ($N=275$) vs. Full ($N=1000$)?** | Reported separate tables for Covered queries and Full test set. | 725 test queries have zero matching scaffolds in the 11,000-candidate corpus. Their true score is mathematically zero. Combining them hides true model ranking quality behind 72.5% zero-inflation. | Proved that full-test MAP is bounded by 0.2750; covered evaluation isolates true retrieval quality. |

---

## 12. Comprehensive Viva Defense & Oral Exam Question Bank (25 Questions & Answers)

### Category A: Problem Formulation & Chemoinformatics

#### Q1: What is the formal computational complexity of exact molecular subgraph matching, and why can't we use it for database search?
**Answer:** Subgraph isomorphism is NP-complete (Ullmann, 1976). Exact algorithms like Ullmann and VF2 exhibit exponential worst-case time complexity $\mathcal{O}(|\mathcal{V}_t|^{|\mathcal{V}_q|} \cdot |\mathcal{V}_q|)$. For a database of millions of molecules, performing pairwise combinatorial backtracking per query would take days. In contrast, neural embeddings compute a single forward pass in polynomial time $\mathcal{O}(|\mathcal{V}| + |\mathcal{E}|)$ and search vector indices in sub-linear time.

#### Q2: How does the Extended-Connectivity Fingerprint (ECFP4) work?
**Answer:** ECFP4 uses the Morgan algorithm with radius $r=2$. At radius 0, it initializes 32-bit hashes of atom invariants (atomic number, valence, charge, aromaticity). In iterations 1 and 2, it collects neighbor tuples (bond order, neighbor hash), sorts them deterministically for isomorphism invariance, and hashes them. All unique integer hashes are collected into a shingle set $\mathcal{S}(\mathcal{G})$ and folded modulo 1024 into a fixed-width binary vector.

#### Q3: Why does ECFP4 fail on continuous structural similarity? Give a concrete chemical example.
**Answer:** ECFP4 treats bit positions as discrete, orthogonal flags. If we substitute a benzene ring with a pyridine ring (a common bioisosteric replacement where a CH is replaced by N), this single change alters the central atom hash at $r=0$, all 1-hop neighbor tuples at $r=1$, and all 2-hop tuples at $r=2$. Multiple bit positions change simultaneously, causing a sharp drop in Tanimoto similarity despite near-identical molecular shape and biological reactivity.

#### Q4: What is a Bemis-Murcko scaffold, and how is it used as ground truth in this project?
**Answer:** A Bemis-Murcko scaffold (Bemis & Murcko, 1996) separates a drug-like molecule into ring systems, linking chains, and terminal side chains. It strips all terminal side chains while preserving all ring assemblies and the linkers connecting them. In this project, a retrieved corpus molecule $c$ is defined as relevant to query $q$ if and only if $\operatorname{BM}(q) = \operatorname{BM}(c)$ and $\operatorname{BM}(q) \neq \text{""}$. Acyclic molecules have an empty scaffold `""` and receive relevance 0 to prevent trivial matches.

---

### Category B: Graph Neural Networks & Expressive Power

#### Q5: What is the 1-Weisfeiler-Lehman (1-WL) test, and why is it important for GNNs?
**Answer:** The 1-WL test is a graph color-refinement algorithm. At each step, a node's color is updated by hashing its current color with the multiset of its neighbor colors. Morris et al. (2019) and Xu et al. (2019) proved that standard message-passing GNNs have at most the discriminative power of the 1-WL test. A GNN can only match 1-WL power if its multiset aggregation function is mathematically injective.

#### Q6: Why does mean or max pooling in GNNs fail multiset injectivity?
**Answer:** Multisets contain elements with multiplicity (e.g., $\{\!\{a, a\}\!\}$ vs. $\{\!\{a, a, a\}\!\}$). Mean pooling yields $a$ for both, losing graph size and node count. Max pooling yields $\max(a, b)$ for $\{\!\{a, b\}\!\}$ and $\{\!\{a, b, b\}\!\}$, ignoring frequency. Sum pooling preserves both element types and counts, making it injective over countable multisets.

#### Q7: Describe your GIN message and update formulation. What activation is used where?
**Answer:** We implement GINEConv with directional bond embeddings matching `src/gin_encoder.py`:
* Directional edge message: $m_{uv}^{(l)} = \operatorname{ReLU}(h_u^{(l-1)} + \boldsymbol{W}_e^{(l)} \boldsymbol{e}_{uv})$, where $\boldsymbol{W}_e \in \mathbb{R}^{64 \times 64}$.
* Multiset aggregation and update: $h_v^{(l)} = \operatorname{MLP}^{(l)}\big( (1 + \epsilon^{(l)}) h_v^{(l-1)} + \sum_{u \in \mathcal{N}(v)} m_{uv}^{(l)} \big)$.
* The activation on the edge-message combination is **$\operatorname{ReLU}$**.
* The internal 2-layer MLP expands to 128 dimensions:
  $$\operatorname{Linear}(64 \to 128) \to \operatorname{BatchNorm1d}(128) \to \operatorname{GELU} \to \operatorname{Linear}(128 \to 64) \to \operatorname{BatchNorm1d}(64)$$
* Each GINE layer applies a residual connection: $h \leftarrow \operatorname{ReLU}(h + h_{\text{new}})$ with $\operatorname{Dropout}(p=0.2)$.
* The projection head maps the $128$-D pooled vector to $256$-D via:
  $$\operatorname{Linear}(128 \to 256) \to \operatorname{LayerNorm}(256) \to \operatorname{GELU} \to \operatorname{Dropout}(0.2) \to \operatorname{Linear}(256 \to 256)$$
  followed by strict $L_2$ normalization onto $\mathbb{S}^{255}$.

#### Q8: Why do you use dual global pooling (sum + mean) instead of just sum pooling?
**Answer:** Sum pooling is an *extensive* readout that captures total atom counts and molecular size. Mean pooling is an *intensive* readout that captures atom density and structural composition independent of size. Concatenating both into a 128-D vector allows the downstream projection head to evaluate both molecular weight and topological density simultaneously.

---

### Category C: Hyperspherical Geometry & Vector Retrieval

#### Q9: Prove why inner product ranking on the unit hypersphere $\mathbb{S}^{255}$ is identical to Euclidean nearest neighbor search.
**Answer:** For unit vectors $\boldsymbol{q}, \boldsymbol{c} \in \mathbb{S}^{255}$, $\|\boldsymbol{q}\|_2^2 = 1$ and $\|\boldsymbol{c}\|_2^2 = 1$. The squared Euclidean distance expands as:
$$\|\boldsymbol{q} - \boldsymbol{c}\|_2^2 = \|\boldsymbol{q}\|_2^2 + \|\boldsymbol{c}\|_2^2 - 2 \langle \boldsymbol{q}, \boldsymbol{c} \rangle = 2 - 2 \langle \boldsymbol{q}, \boldsymbol{c} \rangle$$
Since minimizing $2 - 2\langle \boldsymbol{q}, \boldsymbol{c} \rangle$ is strictly monotonic with maximizing $\langle \boldsymbol{q}, \boldsymbol{c} \rangle$, ranking candidates by cosine similarity is mathematically isomorphic to ranking by Euclidean distance.

#### Q10: Why did you use FAISS IndexFlatIP rather than an approximate index like HNSW for your main results?
**Answer:** The evaluated candidate corpus contains 11,000 molecules. On this corpus, FAISS `IndexFlatIP` performs exhaustive inner-product scanning across 256-D vectors in under 1 millisecond ($0.89\text{ ms}$) on CPU with **zero approximation error (100% recall relative to exact ranking)**. HNSW is implemented in the codebase as an exploratory feature for future multi-million molecule scaling, but using FlatIP ensures our benchmark results reflect pure representation quality without ANN approximation noise.

---

### Category D: Taxonomy, Strategy B, and Metric Learning

#### Q11: What was Strategy A, why was it flawed, and how does Strategy B fix it?
**Answer:** Strategy A extracted scaffolds and assigned class labels based on frequency across the *entire 12,000-molecule dataset*, causing data leakage because test set distributions influenced training cluster definitions. Strategy B fixes this by constructing the taxonomy **strictly from the 10,000 training molecules**. Only scaffolds with $\ge 3$ training occurrences formed classes (241 classes, 1,839 molecules), plus 3 auxiliary acyclic pseudo-classes (55 molecules), while 8,106 rare molecules were isolated with label -1.

#### Q12: Why did you divide the 55 acyclic molecules into 3 pseudo-classes?
**Answer:** 55 training molecules have an empty Murcko scaffold `""`. If grouped into a single class, contrastive loss would falsely pull completely different linear molecules (e.g., hexane and a linear peptide) together. If discarded, valid training data would be lost. Round-robin assignment into 3 auxiliary pseudo-classes ($y_i = 241 + (i \bmod 3)$) allows them to provide supervised training signals without creating a massive false-positive cluster. They are excluded from ground-truth scaffold evaluation.

#### Q13: Explain Supervised Contrastive Loss (SupCon). Why did it achieve the highest MAP@10 (0.5996)?
**Answer:** SupCon pulls all molecules of the same scaffold class together while pushing molecules of all other classes apart, scaled by temperature $\tau = 0.07$. The gradient on negative impostors scales inversely with $\tau$, creating intense repulsive forces against hard negatives. This carves out extremely tight, well-separated hyperspherical clusters, outperforming cross-entropy (+147.5% relative gain) and classical ECFP4 (+161.4% relative gain).

#### Q14: Why did GraphCL augmentations (SupCon-Aug) cut performance in half (from 0.5996 down to 0.2928)?
**Answer:** GraphCL randomly drops bonds ($p=0.1$) and masks atoms ($p=0.1$). In chemistry, covalent bonds constitute the Murcko scaffold. Cleaving a single bond in a benzene or cyclohexyl ring breaks the ring into an open chain, destroying the canonical Murcko scaffold. SupCon-Aug was forcing the model to pull together graph pairs that had different scaffold identities, injecting contradictory supervisory noise directly into training.

#### Q15: What is ProxyAnchor loss, and why did you experiment with it?
**Answer:** In molecular datasets, scaffold frequencies follow a power-law distribution. Mini-batches often lack positive pairs for rare classes, causing contrastive loss to provide zero positive gradient. ProxyAnchor introduces learnable proxy vectors $\boldsymbol{p}_k \in \mathbb{S}^{255}$ for all classes. Every sample is pulled toward its class proxy and repelled from negative proxies, guaranteeing positive gradients regardless of batch composition. It achieved 0.5593 MAP@10.

#### Q16: How do label -1 (rare) molecules behave in ProxyAnchor Neg-Repeller vs. Excluded?
**Answer:** In ProxyAnchor Excluded, label -1 molecules are completely ignored during loss calculation. In Neg-Repeller, label -1 molecules pull no positive proxy, but act as negative repellers across *all 244 class proxies*. Excluded achieved 0.5593 MAP@10, outperforming Neg-Repeller (0.5418), because forcing 8,106 rare molecules to repel all proxies introduced diffuse gradient noise that compressed inter-class margins.

---

### Category E: Evaluation Methodology & Results

#### Q17: Why is full-test MAP@10 bounded by 0.2750? Prove it.
**Answer:** Out of 1,000 test queries, exactly 725 have scaffolds that appear zero times in the 11,000-molecule corpus. For these 725 queries, there are no true positives in the candidate library, so their AP@10 is mathematically 0.0 for any system. The maximum possible MAP@10 across all 1,000 queries is:
$$\max \operatorname{MAP}_{\text{full}} = \frac{275 \times 1.0 + 725 \times 0.0}{1000} = 0.2750$$
Any metric reported on the full test set is scaled down by this 27.5% coverage bound.

#### Q18: What are your authoritative results on Covered Queries ($N = 275$)?
**Answer:**
* **GIN-SupCon (Winner):** MAP@10 = **0.5996**, Recall@10 = **0.5939**
* **GIN-ProxyAnchor (Excluded):** MAP@10 = **0.5593**, Recall@10 = **0.5244**
* **GIN-ProxyAnchor (Neg-Repeller):** MAP@10 = **0.5418**, Recall@10 = **0.5127**
* **GIN-SupCon-Aug:** MAP@10 = **0.2928**, Recall@10 = **0.3663**
* **GIN-CE:** MAP@10 = **0.2423**, Recall@10 = **0.1911**
* **ECFP4 Exact Tanimoto (Baseline):** MAP@10 = **0.2294**, Recall@10 = **0.3116**
* **ECFP4 Exact Shingle Jaccard:** MAP@10 = **0.2304**, Recall@10 = **0.3109**

#### Q19: What is the relative improvement of GIN-SupCon over classical ECFP4?
**Answer:** On covered queries:
$$\frac{0.5996 - 0.2294}{0.2294} \times 100\% = \mathbf{+161.4\% \text{ relative gain in MAP}@10}$$

#### Q20: Why did GIN-CE perform worse than ECFP4 in Recall@10 (0.1911 vs. 0.3116)?
**Answer:** Cross-entropy only learns separating hyperplanes to satisfy multi-class classification on the 244 training classes. It does not enforce compact metric margins. When tested on retrieval where nearest neighbors dictate ranking, samples from the same class scatter widely within their classification half-space, leading to poor recall.

---

### Category F: Methodological Rigor & Viva Traps

#### Q21: Did you evaluate multi-seed statistical significance for your Strategy B winner?
**Answer:** No. In strict adherence to scientific transparency, we disclose that all reported Strategy B benchmark metrics originate from single-run evaluations on fixed saved checkpoints. The multi-seed script in the repository trains a different objective (Neg-Repeller) and cannot be cited to claim statistical stability for the Excluded or SupCon models.

#### Q22: How were checkpoints selected during training?
**Answer:** Models were trained for a fixed 20 epochs using the Adam optimizer. The saved checkpoints were the final epoch-20 weights; validation loss checkpoint selection was not implemented in the training loop. We disclose this as a limitation in Section 3.4 of our report.

#### Q23: Can your GNN model replace bioactivity screening in real pharmaceutical drug discovery?
**Answer:** No. Our model evaluates **2D Bemis-Murcko topological scaffold equivalence**. In biological drug discovery, bioactivity depends on 3D conformational energetics, stereochemistry, electrostatic surface potentials, and binding pocket kinetics. Two molecules with the same Murcko scaffold can have vastly different binding affinities if their functional groups differ, and molecules with different scaffolds can bind the same target (scaffold hopping). Our model is designed for rapid structural candidate filtering, not bioactivity prediction.

#### Q24: What future extension would you prioritize to improve GIN retrieval?
**Answer:** **Chemistry-aware graph augmentations**. Instead of random bond dropping (which destroys ring scaffolds), we should implement motif-preserving augmentations: side-chain variation, terminal group bioisosteric substitutions, or bond perturbations restricted strictly to acyclic linkers, leaving the core cyclic Murcko scaffold intact.

#### Q25: How does your system serve queries in a production deployment?
**Answer:** Offline, candidate molecules are encoded through the 4-layer GIN into 256-D hyperspherical embeddings and indexed in FAISS (`IndexFlatIP`). Online, a new SMILES string is converted to a PyG graph in $\sim 0.8\text{ ms}$, passed through the GIN encoder in $\sim 0.5\text{ ms}$, and searched against FAISS in $\sim 0.89\text{ ms}$, delivering top-10 structural analogs in approximately **2.2 milliseconds total end-to-end latency**.

---
*End of Comprehensive Theoretical Foundations and Viva Guide.*
