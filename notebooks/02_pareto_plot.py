"""
02_pareto_plot.py - Quality vs. Latency Pareto Frontier Visualization
CS328: Introduction to Data Science
Author: Laxmidhar Panda (24110185)

Plots the full multi-pipeline ablation results (MAP@10 vs. Latency) to identify Pareto-optimal architectures.
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np


def plot_pareto_frontier(
    json_path: str = "artifacts/ablation_results.json",
    output_path: str = "artifacts/pareto_frontier.png"
):
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found. Run ablation.py first.")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    pipelines = data["pipelines"]
    names = []
    latencies = []
    maps = []
    recalls = []

    palette = [
        ("#d95f02", "s"),   # ECFP + MinHash (b=32, r=4)
        ("#e6ab02", "v"),   # ECFP + MinHash (b=64, r=2)
        ("#7570b3", "^"),   # ECFP + Binary HNSW
        ("#386cb0", "D"),   # GIN-CE + LSH
        ("#4ea8de", "o"),   # GIN-CE + HNSW
        ("#e7298a", "D"),   # GIN-SupCon + LSH
        ("#1b9e77", "o"),   # GIN-SupCon + HNSW
        ("#66a61e", "P"),   # GIN-SupCon + HNSW + Reranker
        ("#17c3b2", "*"),   # GIN-SupCon-Aug + HNSW
        ("#e63946", "X"),   # GIN-ProxyAnchor + HNSW
    ]

    for p in pipelines:
        names.append(p["name"])
        latencies.append(p["latency_mean_ms"])
        maps.append(p["map_at_10"])
        recalls.append(p["recall_at_10"])

    latencies = np.array(latencies)
    maps = np.array(maps)

    plt.figure(figsize=(11, 7), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    for i, name in enumerate(names):
        col, mark = palette[i % len(palette)]
        plt.scatter(
            latencies[i], maps[i],
            color=col, marker=mark, s=180, edgecolors="black", linewidths=1.2,
            label=f"{name} (MAP: {maps[i]:.3f}, {latencies[i]:.2f} ms)",
            zorder=5
        )

        # Annotations for key points
        if "ProxyAnchor" in name or "SupCon + HNSW" in name or "b=32" in name:
            plt.annotate(
                f" {name}\n MAP: {maps[i]:.3f}\n {latencies[i]:.2f} ms",
                (latencies[i], maps[i]),
                textcoords="offset points",
                xytext=(8, -8),
                fontsize=8.5,
                fontweight="semibold",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=col, alpha=0.85),
                zorder=6
            )

    # Determine Pareto frontier points (maximize MAP, minimize Latency)
    sorted_indices = np.argsort(latencies)
    pareto_idx = []
    max_map_seen = -1.0
    for idx in sorted_indices:
        if maps[idx] > max_map_seen:
            pareto_idx.append(idx)
            max_map_seen = maps[idx]

    pareto_latencies = latencies[pareto_idx]
    pareto_maps = maps[pareto_idx]
    plt.plot(
        pareto_latencies, pareto_maps,
        color="#1d3557", linestyle="--", linewidth=2.0, alpha=0.85,
        label="Pareto Optimal Frontier", zorder=4
    )

    plt.title("Quality vs. Latency Pareto Frontier (ZINC 12k Retrieval)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Mean Single-Query Latency (ms) [Log Scale, Lower is Better]", fontsize=11, fontweight="medium")
    plt.ylabel("Retrieval Quality (MAP@10) [Higher is Better]", fontsize=11, fontweight="medium")
    plt.xscale("log")
    plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8.5, frameon=True)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved expanded Pareto Frontier plot to: {output_path}")


if __name__ == "__main__":
    plot_pareto_frontier()
