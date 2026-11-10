"""
ablation.py - Comprehensive Multi-Depth Factorial Ablation Harness & Benchmark Suite
CS328: Introduction to Data Science
Author: Laxmidhar Panda (24110185)

Directly implements the evaluation protocol of CS_328_Project_Report_previous.pdf:
Measures MAP@1, MAP@5, MAP@10, MAP@50, Recall@1, Recall@5, Recall@10, Recall@50,
MRR, Acc@10 (majority vote family prediction), and single-query Latency across 10 pipelines:

1.  ECFP + MinHash (b=32, r=4)       (Standard Discrete LSH Baseline)
2.  ECFP + MinHash (b=64, r=2)       (Relaxed Threshold Discrete LSH)
3.  ECFP + Binary HNSW               (Discrete Proximity Graph)
4.  GIN-CE + SimHash                 (Cross-Entropy + Hyperplane LSH)
5.  GIN-CE + FAISS HNSW              (Cross-Entropy + Proximity Graph)
6.  GIN-SupCon + SimHash             (Contrastive + Hyperplane LSH)
7.  GIN-SupCon + FAISS HNSW          (Main Proposed Framework)
8.  GIN-SupCon-Aug + FAISS HNSW      (Contrastive + GraphCL Augmentations + Class-Balanced Sampling)
9.  GIN-ProxyAnchor + FAISS HNSW     (Metric Learning via Learnable Class Proxies)
10. GIN-SupCon + HNSW + Reranker     (Two-Stage Precision Retrieval)
"""

import os
import sys
import time
import json
import pickle
from collections import Counter
from typing import List, Dict, Any, Tuple, Set, Optional
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.set_hashing import MinHasher, LSHBandingIndex
from src.vector_search import SimHashIndex, ContinuousHNSWIndex, BinaryHNSWIndex, TwoStageRetriever


def compute_multi_depth_metrics(
    retrieved_indices: List[int],
    ground_truth_relevance: Set[int],
    corpus_labels: np.ndarray,
    query_label: int,
    depths: List[int] = [1, 5, 10, 50]
) -> Dict[str, float]:
    """
    Computes MAP@k, Recall@k across multiple depths, MRR, and Acc@10 (majority vote).
    """
    results = {}
    if not ground_truth_relevance:
        for k in depths:
            results[f"map_{k}"] = 0.0
            results[f"rec_{k}"] = 0.0
        results["mrr"] = 0.0
        results["acc_10"] = 0.0
        return results

    max_k = max(depths)
    top_candidates = retrieved_indices[:max_k]
    hits = [1 if idx in ground_truth_relevance else 0 for idx in top_candidates]

    # Compute MAP and Recall for each depth k
    for k in depths:
        sub_hits = hits[:k]
        num_hits = 0
        prec_sum = 0.0
        for rank, hit in enumerate(sub_hits, start=1):
            if hit:
                num_hits += 1
                prec_sum += (num_hits / rank)
        denom = min(k, len(ground_truth_relevance))
        results[f"map_{k}"] = (prec_sum / denom) if denom > 0 else 0.0
        results[f"rec_{k}"] = (num_hits / len(ground_truth_relevance))

    # MRR (first hit across all retrieved candidates)
    rr = 0.0
    for rank, hit in enumerate(hits, start=1):
        if hit:
            rr = 1.0 / rank
            break
    results["mrr"] = rr

    # Acc@10: Majority vote among top-10 retrieved labels
    top_10 = top_candidates[:10]
    if top_10:
        retrieved_labels = [int(corpus_labels[idx]) for idx in top_10]
        most_common_label = Counter(retrieved_labels).most_common(1)[0][0]
        results["acc_10"] = 1.0 if most_common_label == int(query_label) else 0.0
    else:
        results["acc_10"] = 0.0

    return results


def benchmark_pipeline(
    name: str,
    query_fn,
    num_queries: int,
    ground_truth_fn,
    corpus_labels: np.ndarray,
    query_labels: np.ndarray,
    warmup: int = 50,
    depths: List[int] = [1, 5, 10, 50]
) -> Dict[str, Any]:
    print(f"\n--- Benchmarking {name} ({num_queries} queries) ---")

    # Warmup
    for q_idx in range(min(warmup, num_queries)):
        _ = query_fn(q_idx)

    latencies_ms = []
    accumulators = {f"map_{k}": [] for k in depths}
    accumulators.update({f"rec_{k}": [] for k in depths})
    accumulators["mrr"] = []
    accumulators["acc_10"] = []

    for q_idx in range(num_queries):
        t0 = time.perf_counter()
        top_indices = query_fn(q_idx)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

        gt = ground_truth_fn(q_idx)
        m = compute_multi_depth_metrics(top_indices, gt, corpus_labels, query_labels[q_idx], depths)

        for k in depths:
            accumulators[f"map_{k}"].append(m[f"map_{k}"])
            accumulators[f"rec_{k}"].append(m[f"rec_{k}"])
        accumulators["mrr"].append(m["mrr"])
        accumulators["acc_10"].append(m["acc_10"])

    lat_arr = np.array(latencies_ms)
    result = {
        "name": name,
        "map_at_1": round(float(np.mean(accumulators["map_1"])), 4),
        "map_at_5": round(float(np.mean(accumulators["map_5"])), 4),
        "map_at_10": round(float(np.mean(accumulators["map_10"])), 4),
        "map_at_50": round(float(np.mean(accumulators["map_50"])), 4),
        "recall_at_1": round(float(np.mean(accumulators["rec_1"])), 4),
        "recall_at_5": round(float(np.mean(accumulators["rec_5"])), 4),
        "recall_at_10": round(float(np.mean(accumulators["rec_10"])), 4),
        "recall_at_50": round(float(np.mean(accumulators["rec_50"])), 4),
        "mrr": round(float(np.mean(accumulators["mrr"])), 4),
        "acc_at_10": round(float(np.mean(accumulators["acc_10"])), 4),
        "latency_mean_ms": round(float(np.mean(lat_arr)), 3),
        "latency_std_ms": round(float(np.std(lat_arr)), 3),
        "latency_p50_ms": round(float(np.percentile(lat_arr, 50)), 3),
        "latency_p95_ms": round(float(np.percentile(lat_arr, 95)), 3),
        "latency_p99_ms": round(float(np.percentile(lat_arr, 99)), 3),
    }

    print(f"Results for {name}:")
    print(f"  MAP@10    : {result['map_at_10']:.4f}  |  MAP@50   : {result['map_at_50']:.4f}")
    print(f"  Recall@10 : {result['recall_at_10']:.4f}  |  Recall@50: {result['recall_at_50']:.4f}")
    print(f"  MRR       : {result['mrr']:.4f}  |  Acc@10   : {result['acc_at_10']:.4f}")
    print(f"  Latency   : {result['latency_mean_ms']:.3f} ms (P95: {result['latency_p95_ms']:.3f} ms)")
    return result


def run_full_ablation(
    cache_path: str = "data/processed_cache.pkl",
    output_json: str = "artifacts/ablation_results.json"
) -> Dict[str, Any]:
    print("==========================================================================")
    print(" Master Factorial Ablation Harness (Multi-Depth Evaluation Across 10 Pipelines)")
    print("==========================================================================")

    with open(cache_path, "rb") as f:
        cache = pickle.load(f)

    corpus_shingles = cache["corpus_shingles"]
    query_shingles = cache["query_shingles"]
    corpus_bits = cache["corpus_bits"]
    query_bits = cache["query_bits"]
    corpus_labels = cache["corpus_labels"]
    query_labels = cache["query_labels"]

    num_queries = len(query_labels)
    n_corpus = len(corpus_labels)

    label_to_corpus = {}
    for idx, lbl in enumerate(corpus_labels):
        if lbl not in label_to_corpus:
            label_to_corpus[lbl] = set()
        label_to_corpus[lbl].add(idx)

    def get_gt(q_idx: int) -> Set[int]:
        return label_to_corpus.get(query_labels[q_idx], set())

    pipelines_results = []

    # 1. Pipeline 1: ECFP + MinHash (b=32, r=4)
    print("\n[Indexing] 1. ECFP + MinHash (b=32, r=4)...")
    hasher = MinHasher(128, seed=42)
    c_sig = hasher.compute_signature_matrix(corpus_shingles)
    q_sig = hasher.compute_signature_matrix(query_shingles)
    lsh_32_4 = LSHBandingIndex(b=32, r=4)
    lsh_32_4.index(c_sig)
    p1 = benchmark_pipeline(
        "ECFP + MinHash (b=32, r=4)",
        lambda q: [doc for doc, _ in lsh_32_4.query_top_k(q_sig[:, q], k=50)],
        num_queries, get_gt, corpus_labels, query_labels
    )
    pipelines_results.append(p1)

    # 2. Pipeline 2: ECFP + MinHash (b=64, r=2) (Relaxed threshold)
    print("\n[Indexing] 2. ECFP + MinHash (b=64, r=2, relaxed threshold)...")
    hasher_128_64 = MinHasher(128, seed=42)
    lsh_64_2 = LSHBandingIndex(b=64, r=2)
    lsh_64_2.index(c_sig)
    p2 = benchmark_pipeline(
        "ECFP + MinHash (b=64, r=2)",
        lambda q: [doc for doc, _ in lsh_64_2.query_top_k(q_sig[:, q], k=50)],
        num_queries, get_gt, corpus_labels, query_labels
    )
    pipelines_results.append(p2)

    # 3. Pipeline 3: ECFP + Binary HNSW
    print("\n[Indexing] 3. ECFP + Binary HNSW...")
    bin_hnsw = BinaryHNSWIndex(1024, M=32)
    bin_hnsw.index(corpus_bits)
    p3 = benchmark_pipeline(
        "ECFP + Binary HNSW",
        lambda q: list(bin_hnsw.search(query_bits[q], k=50)[1]),
        num_queries, get_gt, corpus_labels, query_labels
    )
    pipelines_results.append(p3)

    # 4. Pipeline 4 & 5: GIN-CE (Cross-Entropy Baseline)
    if os.path.exists("artifacts/embs_ce_corpus.npy"):
        print("\n[Indexing] 4 & 5. GIN-CE (Cross-Entropy Baseline)...")
        ce_corpus = np.load("artifacts/embs_ce_corpus.npy")
        ce_query = np.load("artifacts/embs_ce_query.npy")

        ce_simhash = SimHashIndex(d=256, l=128, b=32, r=4, seed=42)
        ce_simhash.index(ce_corpus)
        p4 = benchmark_pipeline(
            "GIN-CE + LSH",
            lambda q: list(ce_simhash.search(ce_query[q], k=50)[1]),
            num_queries, get_gt, corpus_labels, query_labels
        )
        pipelines_results.append(p4)

        ce_hnsw = ContinuousHNSWIndex(d=256, M=32, efConstruction=200, efSearch=64)
        ce_hnsw.index(ce_corpus)
        p5 = benchmark_pipeline(
            "GIN-CE + HNSW",
            lambda q: list(ce_hnsw.search(ce_query[q], k=50)[1]),
            num_queries, get_gt, corpus_labels, query_labels
        )
        pipelines_results.append(p5)

    # 6. Pipeline 6 & 7: GIN-SupCon (Standard Contrastive)
    if os.path.exists("artifacts/corpus_embeddings.npy"):
        print("\n[Indexing] 6 & 7. GIN-SupCon (Standard Contrastive)...")
        supcon_corpus = np.load("artifacts/corpus_embeddings.npy")
        supcon_query = np.load("artifacts/query_embeddings.npy")

        supcon_simhash = SimHashIndex(d=256, l=128, b=32, r=4, seed=42)
        supcon_simhash.index(supcon_corpus)
        p6 = benchmark_pipeline(
            "GIN-SupCon + LSH",
            lambda q: list(supcon_simhash.search(supcon_query[q], k=50)[1]),
            num_queries, get_gt, corpus_labels, query_labels
        )
        pipelines_results.append(p6)

        supcon_hnsw = ContinuousHNSWIndex(d=256, M=32, efConstruction=200, efSearch=64)
        supcon_hnsw.index(supcon_corpus)
        p7 = benchmark_pipeline(
            "GIN-SupCon + HNSW",
            lambda q: list(supcon_hnsw.search(supcon_query[q], k=50)[1]),
            num_queries, get_gt, corpus_labels, query_labels
        )
        pipelines_results.append(p7)

        # 10. Pipeline 10: Two-Stage Reranker (HNSW + Shortlist Reranking)
        print("\n[Indexing] 10. GIN-SupCon + HNSW + Reranker (Two-Stage Retrieval)...")
        two_stage = TwoStageRetriever(supcon_hnsw, shortlist_m=50)
        p10 = benchmark_pipeline(
            "GIN-SupCon + HNSW + Reranker",
            lambda q: list(two_stage.search(supcon_query[q], k=50)[1]),
            num_queries, get_gt, corpus_labels, query_labels
        )
        pipelines_results.append(p10)

    # 8. Pipeline 8: GIN-SupCon-Aug (Augmentations + Class-Balanced Sampling)
    if os.path.exists("artifacts/embs_supcon_aug_corpus.npy"):
        print("\n[Indexing] 8. GIN-SupCon-Aug (Augmentations + Balanced Sampling)...")
        aug_corpus = np.load("artifacts/embs_supcon_aug_corpus.npy")
        aug_query = np.load("artifacts/embs_supcon_aug_query.npy")
        aug_hnsw = ContinuousHNSWIndex(d=256, M=32, efConstruction=200, efSearch=64)
        aug_hnsw.index(aug_corpus)
        p8 = benchmark_pipeline(
            "GIN-SupCon-Aug + HNSW",
            lambda q: list(aug_hnsw.search(aug_query[q], k=50)[1]),
            num_queries, get_gt, corpus_labels, query_labels
        )
        pipelines_results.append(p8)

    # 9. Pipeline 9: GIN-ProxyAnchor
    if os.path.exists("artifacts/embs_proxyanchor_corpus.npy"):
        print("\n[Indexing] 9. GIN-ProxyAnchor (Class Proxies)...")
        pa_corpus = np.load("artifacts/embs_proxyanchor_corpus.npy")
        pa_query = np.load("artifacts/embs_proxyanchor_query.npy")
        pa_hnsw = ContinuousHNSWIndex(d=256, M=32, efConstruction=200, efSearch=64)
        pa_hnsw.index(pa_corpus)
        p9 = benchmark_pipeline(
            "GIN-ProxyAnchor + HNSW",
            lambda q: list(pa_hnsw.search(pa_query[q], k=50)[1]),
            num_queries, get_gt, corpus_labels, query_labels
        )
        pipelines_results.append(p9)

    master_results = {
        "dataset": "ZINC 12k Subset",
        "num_corpus": n_corpus,
        "num_queries": num_queries,
        "pipelines": pipelines_results
    }

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(master_results, f, indent=2)

    print(f"\nAll multi-depth ablation results saved to: {output_json}")
    return master_results


if __name__ == "__main__":
    run_full_ablation()
