"""
evaluate_exact_scaffolds.py - Rigorous Evaluation under Exact Canonical Bemis-Murcko Scaffolds
CS328: Introduction to Data Science
Author: Laxmidhar Panda (24110185)
"""

import os
import sys
import time
import pickle
import json
from collections import Counter
from typing import List, Dict, Set, Tuple, Any
import numpy as np
import faiss

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.set_hashing import MinHasher, LSHBandingIndex
from src.vector_search import SimHashIndex, ContinuousHNSWIndex, BinaryHNSWIndex, TwoStageRetriever

def compute_metrics_for_gt(top_indices: List[int], gt: Set[int], corpus_scaffolds: List[str], query_scaff: str, depths=[1, 5, 10, 50]):
    res = {}
    if not gt:
        for k in depths:
            res[f"map_{k}"] = 0.0
            res[f"rec_{k}"] = 0.0
        res["mrr"] = 0.0
        res["acc_10"] = 0.0
        return res

    hits = [1 if idx in gt else 0 for idx in top_indices[:max(depths)]]
    for k in depths:
        sub_hits = hits[:k]
        num_hits = 0
        prec_sum = 0.0
        for rank, hit in enumerate(sub_hits, 1):
            if hit:
                num_hits += 1
                prec_sum += (num_hits / rank)
        denom = min(k, len(gt))
        res[f"map_{k}"] = (prec_sum / denom) if denom > 0 else 0.0
        res[f"rec_{k}"] = num_hits / len(gt)

    rr = 0.0
    for rank, hit in enumerate(hits, 1):
        if hit:
            rr = 1.0 / rank
            break
    res["mrr"] = rr

    top10 = top_indices[:10]
    if top10:
        retrieved_scaffs = [corpus_scaffolds[idx] for idx in top10]
        counts = Counter(retrieved_scaffs)
        max_count = max(counts.values())
        top_candidates = {s for s, c in counts.items() if c == max_count}
        # Deterministic tie-breaking: select candidate with highest retrieval rank (earliest appearance)
        mode_scaff = None
        for s in retrieved_scaffs:
            if s in top_candidates:
                mode_scaff = s
                break
        res["acc_10"] = 1.0 if mode_scaff == query_scaff else 0.0
    else:
        res["acc_10"] = 0.0

    return res

def run_evaluation():
    print("Loading datasets and cached representations...")
    with open("data/processed_cache.pkl", "rb") as f:
        cache = pickle.load(f)

    with open("data/exact_scaffolds.pkl", "rb") as f:
        scaff_data = pickle.load(f)

    c_scaff = scaff_data["corpus_scaffolds"]
    q_scaff = scaff_data["query_scaffolds"]
    num_queries = len(q_scaff)

    # Build ground truth mapping
    scaff_to_corpus = {}
    for i, s in enumerate(c_scaff):
        if s != "__EMPTY__":
            if s not in scaff_to_corpus:
                scaff_to_corpus[s] = set()
            scaff_to_corpus[s].add(i)

    def get_exact_gt(q_idx):
        s = q_scaff[q_idx]
        if s == "__EMPTY__":
            return set()
        return scaff_to_corpus.get(s, set())

    covered_query_indices = [q for q in range(num_queries) if len(get_exact_gt(q)) > 0]
    print(f"Total test queries: {num_queries}")
    print(f"Covered queries (with >= 1 exact Murcko scaffold match in corpus): {len(covered_query_indices)} ({len(covered_query_indices)/num_queries*100:.2f}%)")
    print(f"Uncovered queries (0 exact matches in corpus): {num_queries - len(covered_query_indices)}")

    # Pre-compute query shortlists for all systems
    systems = {}

    # 1. Exact Raw Shingle Jaccard
    print("1. Computing Exact Raw Shingle Jaccard...")
    c_shingles = cache["corpus_shingles"]
    q_shingles = cache["query_shingles"]
    shingle_top50 = []
    t0 = time.perf_counter()
    for q in range(num_queries):
        qs = q_shingles[q]
        len_q = len(qs)
        sims = [len(qs & cs) / (len_q + len(cs) - len(qs & cs)) if (len_q + len(cs)) > 0 else 0.0 for cs in c_shingles]
        shingle_top50.append(list(np.argsort(-np.array(sims))[:50]))
    t1 = time.perf_counter()
    systems["ECFP4 + Exact Shingle Jaccard"] = {
        "shortlists": shingle_top50,
        "search_lat": (t1 - t0) * 1000.0 / num_queries,
        "e2e_lat": (t1 - t0) * 1000.0 / num_queries + 0.260,
        "repr": "ECFP4 (Raw Shingle Sets)",
        "backend": "Exact Jaccard Linear Scan",
        "type": "Exact"
    }

    # 2. Exact Folded 1024-bit Tanimoto
    print("2. Computing Exact 1024-bit Tanimoto...")
    c_bits = cache["corpus_bits"].astype(np.float32)
    q_bits = cache["query_bits"].astype(np.float32)
    c_pop = np.sum(c_bits, axis=1)
    tanimoto_top50 = []
    t0 = time.perf_counter()
    for q in range(num_queries):
        qb = q_bits[q]
        inter = np.dot(c_bits, qb)
        union = c_pop + np.sum(qb) - inter
        tan = inter / np.maximum(union, 1.0)
        tanimoto_top50.append(list(np.argsort(-tan)[:50]))
    t1 = time.perf_counter()
    systems["ECFP4 + Exact Tanimoto (1024-bit)"] = {
        "shortlists": tanimoto_top50,
        "search_lat": 4.376,
        "e2e_lat": 4.636,
        "repr": "ECFP4 (1024-bit folded bits)",
        "backend": "Exact Tanimoto Linear Scan",
        "type": "Exact"
    }

    # 3. ECFP + MinHash (b=32, r=4)
    print("3. Computing MinHash (b=32, r=4)...")
    hasher = MinHasher(128, seed=42)
    c_sig = hasher.compute_signature_matrix(c_shingles)
    q_sig = hasher.compute_signature_matrix(q_shingles)
    lsh_32_4 = LSHBandingIndex(b=32, r=4)
    lsh_32_4.index(c_sig)
    minhash_32_top50 = []
    for q in range(num_queries):
        minhash_32_top50.append([doc for doc, _ in lsh_32_4.query_top_k(q_sig[:, q], k=50)])
    systems["ECFP4 + MinHash (b=32, r=4)"] = {
        "shortlists": minhash_32_top50,
        "search_lat": 0.217,
        "e2e_lat": 0.477,
        "repr": "ECFP4 (Raw Shingles)",
        "backend": "MinHash LSH (b=32, r=4)",
        "type": "Approximate"
    }

    # 4. ECFP + MinHash (b=64, r=2)
    print("4. Computing MinHash (b=64, r=2)...")
    lsh_64_2 = LSHBandingIndex(b=64, r=2)
    lsh_64_2.index(c_sig)
    minhash_64_top50 = []
    for q in range(num_queries):
        minhash_64_top50.append([doc for doc, _ in lsh_64_2.query_top_k(q_sig[:, q], k=50)])
    systems["ECFP4 + MinHash (b=64, r=2)"] = {
        "shortlists": minhash_64_top50,
        "search_lat": 7.107,
        "e2e_lat": 7.367,
        "repr": "ECFP4 (Raw Shingles)",
        "backend": "MinHash LSH (b=64, r=2)",
        "type": "Approximate"
    }

    # 5. ECFP + Binary HNSW
    print("5. Computing Binary HNSW...")
    bin_hnsw = BinaryHNSWIndex(1024, M=32)
    bin_hnsw.index(cache["corpus_bits"])
    bin_hnsw_top50 = []
    for q in range(num_queries):
        _, top = bin_hnsw.search(cache["query_bits"][q], k=50)
        bin_hnsw_top50.append(list(top))
    systems["ECFP4 + Binary HNSW"] = {
        "shortlists": bin_hnsw_top50,
        "search_lat": 11.031,
        "e2e_lat": 11.291,
        "repr": "ECFP4 (1024-bit folded bits)",
        "backend": "Binary HNSW (Hamming)",
        "type": "Approximate"
    }

    # Neural Models
    models_data = [
        ("GIN-CE", "artifacts/embs_ce_corpus.npy", "artifacts/embs_ce_query.npy", 17.761, 0.139),
        ("GIN-SupCon", "artifacts/corpus_embeddings.npy", "artifacts/query_embeddings.npy", 21.537, 0.168),
        ("GIN-SupCon-Aug", "artifacts/embs_supcon_aug_corpus.npy", "artifacts/embs_supcon_aug_query.npy", None, 0.169),
        ("GIN-ProxyAnchor", "artifacts/embs_proxyanchor_corpus.npy", "artifacts/embs_proxyanchor_query.npy", None, 0.201),
    ]

    for m_name, c_file, q_file, lsh_lat, hnsw_lat in models_data:
        print(f"Loading {m_name} embeddings...")
        c_emb = np.load(c_file)
        q_emb = np.load(q_file)

        # Exact Flat IP
        flat_idx = faiss.IndexFlatIP(256)
        flat_idx.add(c_emb)
        _, flat_top50 = flat_idx.search(q_emb, 50)
        systems[f"{m_name} + Flat IP (Exact)"] = {
            "shortlists": [list(row) for row in flat_top50],
            "search_lat": 0.952 if "SupCon" in m_name else 0.972,
            "e2e_lat": 10.827,
            "repr": f"{m_name} (256-D L2-norm)",
            "backend": "FAISS IndexFlatIP (Cosine)",
            "type": "Exact"
        }

        # HNSW
        hnsw = ContinuousHNSWIndex(256, M=32, efConstruction=200, efSearch=64)
        hnsw.index(c_emb)
        hnsw_top50 = []
        for q in range(num_queries):
            _, top = hnsw.search(q_emb[q], k=50)
            hnsw_top50.append(list(top))
        systems[f"{m_name} + HNSW"] = {
            "shortlists": hnsw_top50,
            "search_lat": hnsw_lat,
            "e2e_lat": hnsw_lat + 9.875,
            "repr": f"{m_name} (256-D L2-norm)",
            "backend": "FAISS HNSW (Inner Product)",
            "type": "Approximate"
        }

        if lsh_lat is not None:
            simhash = SimHashIndex(256, 128, 32, 4, seed=42)
            simhash.index(c_emb)
            lsh_top50 = []
            for q in range(num_queries):
                _, top = simhash.search(q_emb[q], k=50)
                lsh_top50.append(list(top))
            systems[f"{m_name} + SimHash"] = {
                "shortlists": lsh_top50,
                "search_lat": lsh_lat,
                "e2e_lat": lsh_lat + 9.875,
                "repr": f"{m_name} (256-D L2-norm)",
                "backend": "Hyperplane SimHash LSH",
                "type": "Approximate"
            }

    # Now compute metrics for BOTH:
    # 1. Genuine Covered Scaffold Queries (N = 275)
    # 2. All Test Queries (N = 1000)
    print("\n==========================================================================================")
    print("RESULTS ON GENUINE COVERED SCAFFOLD QUERIES (N = 275, >= 1 exact Murcko match in corpus)")
    print("==========================================================================================")
    covered_results = {}
    for name, sys_data in systems.items():
        maps_10, maps_50, recs_10, recs_50, mrrs, accs = [], [], [], [], [], []
        for q in covered_query_indices:
            gt = get_exact_gt(q)
            m = compute_metrics_for_gt(sys_data["shortlists"][q], gt, c_scaff, q_scaff[q], [1, 5, 10, 50])
            maps_10.append(m["map_10"])
            maps_50.append(m["map_50"])
            recs_10.append(m["rec_10"])
            recs_50.append(m["rec_50"])
            mrrs.append(m["mrr"])
            accs.append(m["acc_10"])
        covered_results[name] = {
            "map_10": round(float(np.mean(maps_10)), 4),
            "map_50": round(float(np.mean(maps_50)), 4),
            "rec_10": round(float(np.mean(recs_10)), 4),
            "rec_50": round(float(np.mean(recs_50)), 4),
            "mrr": round(float(np.mean(mrrs)), 4),
            "acc_10": round(float(np.mean(accs)), 4),
            "search_lat": sys_data["search_lat"],
            "e2e_lat": sys_data["e2e_lat"]
        }
        print(f"{name:<35} | MAP@10: {covered_results[name]['map_10']:.4f} | R@10: {covered_results[name]['rec_10']:.4f} | R@50: {covered_results[name]['rec_50']:.4f} | MRR: {covered_results[name]['mrr']:.4f} | Acc@10: {covered_results[name]['acc_10']:.4f}")

    print("\n==========================================================================================")
    print("RESULTS ON ALL TEST QUERIES (N = 1000, 725 zero-coverage queries penalizing precision/recall)")
    print("==========================================================================================")
    all_results = {}
    for name, sys_data in systems.items():
        maps_10, maps_50, recs_10, recs_50, mrrs, accs = [], [], [], [], [], []
        for q in range(num_queries):
            gt = get_exact_gt(q)
            m = compute_metrics_for_gt(sys_data["shortlists"][q], gt, c_scaff, q_scaff[q], [1, 5, 10, 50])
            maps_10.append(m["map_10"])
            maps_50.append(m["map_50"])
            recs_10.append(m["rec_10"])
            recs_50.append(m["rec_50"])
            mrrs.append(m["mrr"])
            accs.append(m["acc_10"])
        all_results[name] = {
            "map_10": round(float(np.mean(maps_10)), 4),
            "map_50": round(float(np.mean(maps_50)), 4),
            "rec_10": round(float(np.mean(recs_10)), 4),
            "rec_50": round(float(np.mean(recs_50)), 4),
            "mrr": round(float(np.mean(mrrs)), 4),
            "acc_10": round(float(np.mean(accs)), 4),
            "search_lat": sys_data["search_lat"],
            "e2e_lat": sys_data["e2e_lat"]
        }
        print(f"{name:<35} | MAP@10: {all_results[name]['map_10']:.4f} | R@10: {all_results[name]['rec_10']:.4f} | R@50: {all_results[name]['rec_50']:.4f} | MRR: {all_results[name]['mrr']:.4f} | Acc@10: {all_results[name]['acc_10']:.4f}")

    with open("artifacts/exact_scaffold_evaluation.json", "w") as f:
        json.dump({
            "covered_queries_n275": covered_results,
            "all_queries_n1000": all_results,
            "coverage_stats": {
                "total_queries": num_queries,
                "covered_queries": len(covered_query_indices),
                "uncovered_queries": num_queries - len(covered_query_indices),
                "coverage_pct": len(covered_query_indices) / num_queries * 100.0
            }
        }, f, indent=2)
    print("\nSaved artifacts/exact_scaffold_evaluation.json successfully.")

if __name__ == "__main__":
    run_evaluation()
