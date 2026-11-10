"""
03_depth_analysis.py - MAP@k and Recall@k Curves Across Retrieval Depth k in [1, 50]
CS328: Introduction to Data Science
Author: Laxmidhar Panda (24110185)

Matches Figure 3 of CS_328_Project_Report_previous.pdf:
Demonstrates that GIN-based systems maintain strong ranking quality deeper into the shortlist,
whereas discrete ECFP baselines degrade much earlier.
"""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.set_hashing import MinHasher, LSHBandingIndex
from src.vector_search import ContinuousHNSWIndex, BinaryHNSWIndex


def compute_metrics_at_k(retrieved: list, gt: set, k: int):
    top_k = retrieved[:k]
    hits = [1 if idx in gt else 0 for idx in top_k]
    num_hits = 0
    prec_sum = 0.0
    for rank, hit in enumerate(hits, start=1):
        if hit:
            num_hits += 1
            prec_sum += (num_hits / rank)
    denom = min(k, len(gt))
    ap = (prec_sum / denom) if denom > 0 else 0.0
    rec = (num_hits / len(gt)) if len(gt) > 0 else 0.0
    return ap, rec


def run_depth_analysis(output_path: str = "artifacts/depth_curves.png"):
    print("=== Running Multi-Depth Retrieval Analysis (k in [1, 50]) ===")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open("data/processed_cache.pkl", "rb") as f:
        cache = pickle.load(f)

    corpus_labels = cache["corpus_labels"]
    query_labels = cache["query_labels"]
    corpus_bits = cache["corpus_bits"]
    query_bits = cache["query_bits"]

    # Pre-index ground truth
    label_to_corpus = {}
    for i, lbl in enumerate(corpus_labels):
        if lbl not in label_to_corpus:
            label_to_corpus[lbl] = set()
        label_to_corpus[lbl].add(i)

    # 1. Pipeline 1: ECFP + MinHash
    hasher = MinHasher(128, seed=42)
    c_sigs = hasher.compute_signature_matrix(cache["corpus_shingles"])
    q_sigs = hasher.compute_signature_matrix(cache["query_shingles"])
    minhash_index = LSHBandingIndex(b=32, r=4)
    minhash_index.index(c_sigs)

    # 2. Pipeline 2: ECFP + Binary HNSW
    bin_hnsw = BinaryHNSWIndex(1024, M=32)
    bin_hnsw.index(corpus_bits)

    # 3. Pipeline 4: GIN-SupCon + FAISS HNSW
    corpus_embs = np.load("artifacts/corpus_embeddings.npy")
    query_embs = np.load("artifacts/query_embeddings.npy")
    cont_hnsw = ContinuousHNSWIndex(256, M=32, efConstruction=200, efSearch=64)
    cont_hnsw.index(corpus_embs)

    # Optional: GIN-CE if available
    ce_available = os.path.exists("artifacts/embs_ce_corpus.npy")
    if ce_available:
        ce_corpus = np.load("artifacts/embs_ce_corpus.npy")
        ce_query = np.load("artifacts/embs_ce_query.npy")
        ce_hnsw = ContinuousHNSWIndex(256, M=32, efConstruction=200, efSearch=64)
        ce_hnsw.index(ce_corpus)

    depths = [1, 3, 5, 10, 15, 20, 30, 40, 50]
    num_queries = len(query_labels)

    # Precompute top-50 for all systems
    print("Retrieving top-50 shortlists for all queries...")
    p1_results, p2_results, p4_results, ce_results = [], [], [], []

    for q in range(num_queries):
        # P1
        r1 = [doc for doc, _ in minhash_index.query_top_k(q_sigs[:, q], k=50)]
        p1_results.append(r1)

        # P2
        _, r2 = bin_hnsw.search(query_bits[q], k=50)
        p2_results.append(list(r2))

        # P4
        _, r4 = cont_hnsw.search(query_embs[q], k=50)
        p4_results.append(list(r4))

        if ce_available:
            _, rce = ce_hnsw.search(ce_query[q], k=50)
            ce_results.append(list(rce))

    # Evaluate across k
    p1_maps, p1_recs = [], []
    p2_maps, p2_recs = [], []
    p4_maps, p4_recs = [], []
    ce_maps, ce_recs = [], []

    for k in depths:
        m1, r1, m2, r2, m4, r4, mce, rce = [], [], [], [], [], [], [], []
        for q in range(num_queries):
            gt = label_to_corpus.get(query_labels[q], set())
            ap1, rec1 = compute_metrics_at_k(p1_results[q], gt, k)
            ap2, rec2 = compute_metrics_at_k(p2_results[q], gt, k)
            ap4, rec4 = compute_metrics_at_k(p4_results[q], gt, k)
            m1.append(ap1); r1.append(rec1)
            m2.append(ap2); r2.append(rec2)
            m4.append(ap4); r4.append(rec4)
            if ce_available:
                apce, recce = compute_metrics_at_k(ce_results[q], gt, k)
                mce.append(apce); rce.append(recce)

        p1_maps.append(np.mean(m1)); p1_recs.append(np.mean(r1))
        p2_maps.append(np.mean(m2)); p2_recs.append(np.mean(r2))
        p4_maps.append(np.mean(m4)); p4_recs.append(np.mean(r4))
        if ce_available:
            ce_maps.append(np.mean(mce)); ce_recs.append(np.mean(rce))

    # Plot two panels: Left = MAP@k, Right = Recall@k
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Panel 1: MAP@k
    ax1.plot(depths, p4_maps, "o-", color="#1b9e77", linewidth=2.5, label="GIN-SupCon + FAISS HNSW")
    if ce_available:
        ax1.plot(depths, ce_maps, "d-", color="#386cb0", linewidth=2.0, label="GIN-CE + FAISS HNSW")
    ax1.plot(depths, p2_maps, "^-", color="#7570b3", linewidth=2.0, label="ECFP + Binary HNSW")
    ax1.plot(depths, p1_maps, "s-", color="#d95f02", linewidth=2.0, label="ECFP + MinHash")

    ax1.set_title("MAP@k Curves Across Retrieval Depth", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Retrieval Cutoff Depth (k)", fontsize=11)
    ax1.set_ylabel("Mean Average Precision (MAP@k)", fontsize=11)
    ax1.set_xlim(1, 50)
    ax1.legend(loc="upper right", frameon=True, fontsize=10)

    # Panel 2: Recall@k
    ax2.plot(depths, p4_recs, "o-", color="#1b9e77", linewidth=2.5, label="GIN-SupCon + FAISS HNSW")
    if ce_available:
        ax2.plot(depths, ce_recs, "d-", color="#386cb0", linewidth=2.0, label="GIN-CE + FAISS HNSW")
    ax2.plot(depths, p2_recs, "^-", color="#7570b3", linewidth=2.0, label="ECFP + Binary HNSW")
    ax2.plot(depths, p1_recs, "s-", color="#d95f02", linewidth=2.0, label="ECFP + MinHash")

    ax2.set_title("Recall@k Curves Across Retrieval Depth", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Retrieval Cutoff Depth (k)", fontsize=11)
    ax2.set_ylabel("Recall@k (Coverage of Relevant Set)", fontsize=11)
    ax2.set_xlim(1, 50)
    ax2.legend(loc="lower right", frameon=True, fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Depth curves successfully saved to: {output_path}")


if __name__ == "__main__":
    run_depth_analysis()
