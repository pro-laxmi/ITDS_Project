"""
multiseed_strategy_b.py - Train Strategy B ProxyAnchor across multiple seeds (42, 43, 44)
to evaluate statistical reliability, variance, and confidence intervals.
"""

import os
import sys
import pickle
import json
from collections import Counter
import numpy as np
import torch
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
import faiss

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.gin_encoder import GINEncoder
from src.train_strategy_b_objectives import (
    ProxyAnchorWithUnlabeled, build_strategy_b_taxonomy,
    train_proxy_anchor_variant, extract_embeddings_all,
    evaluate_index_system
)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Multi-Seed Reliability Experiment for Strategy B ProxyAnchor on {device}...")

    train_ds = ZINC(root="data/ZINC", subset=True, split="train")
    val_ds = ZINC(root="data/ZINC", subset=True, split="val")
    test_ds = ZINC(root="data/ZINC", subset=True, split="test")

    with open("data/exact_scaffolds.pkl", "rb") as f:
        scaff_data = pickle.load(f)
    c_scaff = scaff_data["corpus_scaffolds"]
    q_scaff = scaff_data["query_scaffolds"]
    train_scaff = c_scaff[:10000]

    scaff_to_corpus = {}
    for i, s in enumerate(c_scaff):
        if s != "__EMPTY__":
            if s not in scaff_to_corpus:
                scaff_to_corpus[s] = set()
            scaff_to_corpus[s].add(i)

    covered_indices = [q for q in range(len(q_scaff)) if q_scaff[q] != "__EMPTY__" and len(scaff_to_corpus.get(q_scaff[q], set())) > 0]

    train_labels_b, _, num_classes_b = build_strategy_b_taxonomy(train_scaff)

    seeds = [42, 43, 44]
    seed_results = []

    for seed in seeds:
        print(f"\n--- Training Strategy B ProxyAnchor with Seed {seed} ---")
        model = train_proxy_anchor_variant(
            train_ds, train_labels_b, num_classes_b, ProxyAnchorWithUnlabeled,
            device, epochs=20, seed=seed, desc=f"GIN-PA-Seed{seed}"
        )
        c_embs, q_embs = extract_embeddings_all(model, train_ds, val_ds, test_ds, device)

        # Exact Flat IP
        flat = faiss.IndexFlatIP(256)
        flat.add(c_embs)
        _, flat_I = flat.search(q_embs, 50)
        res_flat = evaluate_index_system(flat_I, c_scaff, q_scaff, covered_indices)

        # HNSW
        hnsw = faiss.IndexHNSWFlat(256, 32, faiss.METRIC_INNER_PRODUCT)
        hnsw.hnsw.efSearch = 64
        hnsw.add(c_embs)
        _, hnsw_I = hnsw.search(q_embs, 50)
        res_hnsw = evaluate_index_system(hnsw_I, c_scaff, q_scaff, covered_indices)

        ann_rec = float(np.mean([len(set(flat_I[q, :10]) & set(hnsw_I[q, :10])) / 10.0 for q in range(len(q_embs))]))
        res_hnsw["ann_recall_at_10"] = ann_rec

        cov = res_flat["covered"]
        print(f"Seed {seed} Flat IP -> MAP@10: {cov['map_10']:.4f} | R@10: {cov['rec_10']:.4f} | MRR: {cov['mrr']:.4f} | Acc@10: {cov['acc_10']:.4f}")
        seed_results.append({
            "seed": seed,
            "flat_ip": res_flat,
            "hnsw": res_hnsw
        })

    # Summary statistics across seeds
    maps = [r["flat_ip"]["covered"]["map_10"] for r in seed_results]
    recs = [r["flat_ip"]["covered"]["rec_10"] for r in seed_results]
    mrrs = [r["flat_ip"]["covered"]["mrr"] for r in seed_results]
    accs = [r["flat_ip"]["covered"]["acc_10"] for r in seed_results]

    summary = {
        "seeds": seeds,
        "runs": seed_results,
        "covered_mean_std": {
            "map_10": {"mean": float(np.mean(maps)), "std": float(np.std(maps))},
            "rec_10": {"mean": float(np.mean(recs)), "std": float(np.std(recs))},
            "mrr": {"mean": float(np.mean(mrrs)), "std": float(np.std(mrrs))},
            "acc_10": {"mean": float(np.mean(accs)), "std": float(np.std(accs))}
        }
    }

    out_file = "artifacts/multiseed_strategy_b_results.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*80)
    print("Multi-Seed Statistical Summary for Strategy B ProxyAnchor (Exact Flat IP, Covered Scaffolds):")
    print(f"  MAP@10:   {np.mean(maps):.4f} +/- {np.std(maps):.4f} (range: [{min(maps):.4f}, {max(maps):.4f}])")
    print(f"  Recall@10:{np.mean(recs):.4f} +/- {np.std(recs):.4f} (range: [{min(recs):.4f}, {max(recs):.4f}])")
    print(f"  MRR:      {np.mean(mrrs):.4f} +/- {np.std(mrrs):.4f} (range: [{min(mrrs):.4f}, {max(mrrs):.4f}])")
    print(f"  Acc@10:   {np.mean(accs):.4f} +/- {np.std(accs):.4f} (range: [{min(accs):.4f}, {max(accs):.4f}])")
    print("="*80)
    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()
