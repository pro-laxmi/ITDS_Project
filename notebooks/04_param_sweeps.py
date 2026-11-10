"""
04_param_sweeps.py - Hyperparameter Sweeps: HNSW efSearch and MinHash Banding Thresholds
CS328: Introduction to Data Science
Author: Laxmidhar Panda (24110185)
"""

import os
import sys
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.vector_search import ContinuousHNSWIndex
from src.set_hashing import LSHBandingIndex


def sweep_ef_search(corpus_embs: np.ndarray, query_embs: np.ndarray, query_labels: np.ndarray, corpus_labels: np.ndarray):
    print("--- Sweeping FAISS HNSW efSearch in [8, 16, 32, 50, 64, 100, 200] ---")
    ef_values = [8, 16, 32, 50, 64, 100, 200]
    latencies = []
    maps = []

    label_to_corpus = {}
    for i, lbl in enumerate(corpus_labels):
        if lbl not in label_to_corpus:
            label_to_corpus[lbl] = set()
        label_to_corpus[lbl].add(i)

    hnsw = ContinuousHNSWIndex(d=256, M=32, efConstruction=200, efSearch=64)
    hnsw.index(corpus_embs)

    num_queries = min(500, len(query_labels))

    for ef in ef_values:
        hnsw.set_ef_search(ef)
        # Warmup
        for q in range(20):
            _ = hnsw.search(query_embs[q], k=10)

        t_list = []
        ap_list = []
        for q in range(num_queries):
            t0 = time.perf_counter()
            _, top10 = hnsw.search(query_embs[q], k=10)
            t1 = time.perf_counter()
            t_list.append((t1 - t0) * 1000.0)

            gt = label_to_corpus.get(query_labels[q], set())
            hits = [1 if idx in gt else 0 for idx in top10]
            num_hits = 0
            p_sum = 0.0
            for rank, hit in enumerate(hits, 1):
                if hit:
                    num_hits += 1
                    p_sum += (num_hits / rank)
            denom = min(10, len(gt))
            ap_list.append(p_sum / denom if denom > 0 else 0.0)

        mean_lat = float(np.mean(t_list))
        mean_map = float(np.mean(ap_list))
        latencies.append(mean_lat)
        maps.append(mean_map)
        print(f"  efSearch={ef:<3d} | Latency: {mean_lat:.3f} ms | MAP@10: {mean_map:.4f}")

    return ef_values, latencies, maps


def run_sweeps(output_path: str = "artifacts/param_sweeps.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open("data/processed_cache.pkl", "rb") as f:
        cache = pickle.load(f)

    corpus_embs = np.load("artifacts/corpus_embeddings.npy")
    query_embs = np.load("artifacts/query_embeddings.npy")
    corpus_labels = cache["corpus_labels"]
    query_labels = cache["query_labels"]

    ef_vals, lats, map_scores = sweep_ef_search(corpus_embs, query_embs, query_labels, corpus_labels)

    # MinHash theoretical banding curves
    s = np.linspace(0.0, 1.0, 200)
    banding_configs = [
        (16, 8, "#d95f02", "Strict (b=16, r=8, t* ≈ 0.707)"),
        (32, 4, "#1b9e77", "Balanced (b=32, r=4, t* ≈ 0.420)"),
        (64, 2, "#7570b3", "High-Recall (b=64, r=2, t* ≈ 0.125)"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Panel 1: HNSW efSearch vs Latency and MAP
    color_map = "#1b9e77"
    color_lat = "#d95f02"

    ax1.plot(ef_vals, map_scores, "o-", color=color_map, linewidth=2.2, label="MAP@10")
    ax1.set_xlabel("HNSW efSearch Hyperparameter", fontsize=11, fontweight="medium")
    ax1.set_ylabel("Retrieval Quality (MAP@10)", color=color_map, fontsize=11, fontweight="medium")
    ax1.tick_params(axis='y', labelcolor=color_map)

    ax1_twin = ax1.twinx()
    ax1_twin.plot(ef_vals, lats, "s--", color=color_lat, linewidth=2.0, label="Latency (ms)")
    ax1_twin.set_ylabel("Single-Query Latency (ms)", color=color_lat, fontsize=11, fontweight="medium")
    ax1_twin.tick_params(axis='y', labelcolor=color_lat)
    ax1.set_title("HNSW Query Candidate Pool (efSearch) Sensitivity", fontsize=12, fontweight="bold")

    # Panel 2: MinHash S-Curve Phase Transitions
    for b, r, col, label in banding_configs:
        prob = LSHBandingIndex.theoretical_collision_prob(s, b=b, r=r)
        ax2.plot(s, prob, color=col, linewidth=2.2, label=label)
        t_star = (1.0 / b) ** (1.0 / r)
        ax2.axvline(t_star, color=col, linestyle=":", alpha=0.6)

    ax2.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("True Jaccard Similarity J(A, B)", fontsize=11, fontweight="medium")
    ax2.set_ylabel("Candidate Collision Probability P", fontsize=11, fontweight="medium")
    ax2.set_title("LSH MinHash Banding S-Curve Threshold Tuning (k=128)", fontsize=12, fontweight="bold")
    ax2.legend(loc="lower right", fontsize=10, frameon=True)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved parameter sweep figures to: {output_path}")


if __name__ == "__main__":
    run_sweeps()
