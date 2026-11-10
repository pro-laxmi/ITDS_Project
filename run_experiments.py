"""
run_experiments.py - Master Execution Script for 4-Way Factorial Ablation
CS328: Introduction to Data Science
Author: Laxmidhar Panda (24110185)
"""

import os
import sys
import pickle
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src.ablation import run_full_ablation
import importlib.util


def main():
    print("=====================================================================")
    print(" CS328: Scalable Molecular Graph Retrieval - 4-Way Factorial Benchmark")
    print(" Author: Laxmidhar Panda (Roll No: 24110185)")
    print("=====================================================================")

    cache_path = "data/processed_cache.pkl"
    corpus_emb_path = "artifacts/corpus_embeddings.npy"
    query_emb_path = "artifacts/query_embeddings.npy"
    output_json = "artifacts/ablation_results.json"
    pareto_img = "artifacts/pareto_frontier.png"

    assert os.path.exists(cache_path), f"Missing cache: {cache_path}. Run data_ingestion.py first."
    assert os.path.exists(corpus_emb_path), f"Missing embeddings: {corpus_emb_path}. Run train_gin.py first."

    print("Loading precomputed discrete fingerprints and structural labels...")
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)

    corpus_shingles = cache["corpus_shingles"]
    query_shingles = cache["query_shingles"]
    corpus_bits = cache["corpus_bits"]
    query_bits = cache["query_bits"]
    corpus_labels = cache["corpus_labels"]
    query_labels = cache["query_labels"]

    print("Loading precomputed GIN continuous embeddings...")
    corpus_embeddings = np.load(corpus_emb_path)
    query_embeddings = np.load(query_emb_path)

    print(f"Corpus size: {len(corpus_labels)} graphs")
    print(f"Query size : {len(query_labels)} graphs")
    print(f"Embedding dimension: {corpus_embeddings.shape[1]}-D L2-normalized")

    print("\nExecuting 4-way ablation suite...")
    results = run_full_ablation(
        corpus_shingles=corpus_shingles,
        query_shingles=query_shingles,
        corpus_bits=corpus_bits,
        query_bits=query_bits,
        corpus_embeddings=corpus_embeddings,
        query_embeddings=query_embeddings,
        corpus_labels=corpus_labels,
        query_labels=query_labels,
        k=10,
        output_json=output_json
    )

    print("\nGenerating Quality vs. Latency Pareto Frontier plot...")
    import importlib.util
    spec = importlib.util.spec_from_file_location("pareto_plot", "notebooks/02_pareto_plot.py")
    pareto_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pareto_module)
    pareto_module.plot_pareto_frontier(json_path=output_json, output_path=pareto_img)

    # Print final Markdown comparison table
    print("\n" + "="*70)
    print("                     FINAL 4-WAY ABLATION SUMMARY")
    print("="*70)
    print(f"{'Pipeline':<22} | {'MAP@10':<8} | {'Recall@10':<10} | {'MRR':<8} | {'Latency (ms)':<12}")
    print("-" * 70)
    for p in results["pipelines"]:
        print(f"{p['name']:<22} | {p['map_at_10']:<8.4f} | {p['recall_at_10']:<10.4f} | {p['mrr']:<8.4f} | {p['latency_mean_ms']:<6.2f} (±{p['latency_std_ms']:.2f})")
    print("="*70)


if __name__ == "__main__":
    main()
