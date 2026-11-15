"""
02_pareto_plot.py - Quality vs. Latency Pareto Frontier Visualizations (Strategy B Primary + Strategy A Diagnostic)
CS328: Introduction to Data Science
Author: Laxmidhar Panda (24110185)

Generates:
1. artifacts/pareto_frontier_search.png: MAP@10 vs. Pure Index Search Latency
2. artifacts/pareto_frontier_e2e.png: MAP@10 vs. End-to-End CPU Latency
3. artifacts/pareto_frontier.png: Combined 2-Panel Publication Figure
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np


def generate_pareto_plots(output_dir: str = "artifacts"):
    os.makedirs(output_dir, exist_ok=True)

    # 1. System data points: (Name, Representation, Category, MAP@10 on Covered N=275, Search Lat (ms), E2E CPU Lat (ms), color, marker)
    # Reconciled timings: Flat IP = 0.894 ms, HNSW = 0.161 ms, GNN CPU = 9.875 ms, ECFP extraction = 0.260 ms
    systems = [
        # Baselines
        {"name": "ECFP4 + Exact Shingle Jaccard", "cat": "Baseline", "map": 0.2304, "search": 56.42, "e2e": 56.68, "col": "#8c564b", "mark": "p"},
        {"name": "ECFP4 + Exact Tanimoto", "cat": "Baseline", "map": 0.2294, "search": 4.376, "e2e": 4.636, "col": "#9467bd", "mark": "h"},
        {"name": "ECFP4 + MinHash (b=32, r=4)", "cat": "Baseline", "map": 0.1668, "search": 0.217, "e2e": 0.477, "col": "#d95f02", "mark": "s"},
        {"name": "ECFP4 + MinHash (b=64, r=2)", "cat": "Baseline", "map": 0.2198, "search": 7.107, "e2e": 7.367, "col": "#e6ab02", "mark": "v"},
        {"name": "ECFP4 + Binary HNSW", "cat": "Baseline", "map": 0.2251, "search": 11.031, "e2e": 11.291, "col": "#7570b3", "mark": "^"},
        
        # Strategy B Primary Results (Strict Leak-Free Taxonomy)
        {"name": "GIN-CE (Strat B) + Flat IP", "cat": "Strategy B", "map": 0.2423, "search": 0.894, "e2e": 10.769, "col": "#386cb0", "mark": "+"},
        {"name": "GIN-CE (Strat B) + HNSW", "cat": "Strategy B", "map": 0.2423, "search": 0.161, "e2e": 10.036, "col": "#4ea8de", "mark": "o"},
        {"name": "GIN-SupCon-Aug (Strat B) + HNSW", "cat": "Strategy B", "map": 0.2928, "search": 0.161, "e2e": 10.036, "col": "#17c3b2", "mark": "*"},
        {"name": "GIN-ProxyAnchor (Strat B) + Flat IP", "cat": "Strategy B", "map": 0.5593, "search": 0.894, "e2e": 10.769, "col": "#e63946", "mark": "x"},
        {"name": "GIN-ProxyAnchor (Strat B) + HNSW", "cat": "Strategy B", "map": 0.5593, "search": 0.161, "e2e": 10.036, "col": "#e63946", "mark": "X"},
        {"name": "GIN-SupCon (Strat B) + Flat IP", "cat": "Strategy B", "map": 0.5996, "search": 0.894, "e2e": 10.769, "col": "#2ca02c", "mark": "d"},
        {"name": "GIN-SupCon (Strat B) + HNSW", "cat": "Strategy B", "map": 0.5996, "search": 0.161, "e2e": 10.036, "col": "#1b9e77", "mark": "D"},

        # Strategy A Diagnostic (Legacy Leaked Taxonomy)
        {"name": "GIN-ProxyAnchor (Strat A Legacy) + HNSW", "cat": "Diagnostic", "map": 0.6227, "search": 0.161, "e2e": 10.036, "col": "#b2182b", "mark": "P"}
    ]

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # ------------------------------------------------------------------
    # 1. Standalone Search Latency Pareto Plot
    # ------------------------------------------------------------------
    plt.figure(figsize=(10, 6.5), dpi=300)
    for s in systems:
        alpha = 0.5 if s["cat"] == "Diagnostic" else 0.9
        plt.scatter(s["search"], s["map"], color=s["col"], marker=s["mark"], s=140, edgecolors="black", linewidths=1.0, alpha=alpha, zorder=5, label=s["name"])

    # Compute leak-free Pareto frontier (Baselines + Strategy B)
    clean_sys = [s for s in systems if s["cat"] != "Diagnostic"]
    clean_sys.sort(key=lambda x: (x["search"], -x["map"]))
    p_search = []
    max_m = -1.0
    for s in clean_sys:
        if s["map"] > max_m:
            p_search.append(s)
            max_m = s["map"]
    
    xs = [s["search"] for s in p_search]
    ys = [s["map"] for s in p_search]
    if len(xs) > 1:
        plt.plot(xs, ys, color="#1b9e77", linestyle="--", linewidth=2.2, alpha=0.9, label="Pareto Frontier (Leak-Free Strategy B)")
    else:
        plt.scatter(xs, ys, color="#1b9e77", s=250, facecolors='none', edgecolors="#1b9e77", linewidths=2.5, label="Global Optimum: GIN-SupCon + HNSW")

    # Legacy point connection
    plt.plot([0.161, 0.161], [0.5996, 0.6227], color="#b2182b", linestyle=":", linewidth=1.5, alpha=0.7, label="Taxonomy Leakage Delta (+0.0231)")

    plt.xscale("log")
    plt.xlabel("Index Search Latency (ms) [Log Scale, Lower is Better]", fontsize=11, fontweight="bold")
    plt.ylabel("Retrieval Quality (MAP@10 on Exact Covered Scaffolds)", fontsize=11, fontweight="bold")
    plt.title("Quality vs. Index Search Latency Pareto Frontier (Leak-Free Primary)", fontsize=12, fontweight="bold")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pareto_frontier_search.png"))
    plt.close()

    # ------------------------------------------------------------------
    # 2. Standalone End-to-End CPU Latency Pareto Plot
    # ------------------------------------------------------------------
    plt.figure(figsize=(10, 6.5), dpi=300)
    for s in systems:
        alpha = 0.5 if s["cat"] == "Diagnostic" else 0.9
        plt.scatter(s["e2e"], s["map"], color=s["col"], marker=s["mark"], s=140, edgecolors="black", linewidths=1.0, alpha=alpha, zorder=5, label=s["name"])

    # Compute leak-free E2E Pareto frontier
    clean_sys_e2e = [s for s in systems if s["cat"] != "Diagnostic"]
    clean_sys_e2e.sort(key=lambda x: (x["e2e"], -x["map"]))
    p_e2e = []
    max_m = -1.0
    for s in clean_sys_e2e:
        if s["map"] > max_m:
            p_e2e.append(s)
            max_m = s["map"]

    xs_e = [s["e2e"] for s in p_e2e]
    ys_e = [s["map"] for s in p_e2e]
    plt.plot(xs_e, ys_e, color="#e63946", linestyle="--", linewidth=2.2, alpha=0.9, label="Pareto Frontier (E2E CPU, Strategy B)")

    plt.xscale("log")
    plt.xlabel("End-to-End Single-Query CPU Latency (ms) [Log Scale, Lower is Better]", fontsize=11, fontweight="bold")
    plt.ylabel("Retrieval Quality (MAP@10 on Exact Covered Scaffolds)", fontsize=11, fontweight="bold")
    plt.title("Quality vs. End-to-End CPU Latency Pareto Frontier (Including Feature Extraction / Inference)", fontsize=12, fontweight="bold")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pareto_frontier_e2e.png"))
    plt.close()

    # ------------------------------------------------------------------
    # 3. Combined Dual-Panel Publication Figure
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.2), dpi=300)

    # Panel A: Search Latency
    for s in systems:
        alpha = 0.5 if s["cat"] == "Diagnostic" else 0.9
        ax1.scatter(s["search"], s["map"], color=s["col"], marker=s["mark"], s=130, edgecolors="black", linewidths=0.8, alpha=alpha, zorder=5)
    if len(xs) > 1:
        ax1.plot(xs, ys, color="#1b9e77", linestyle="--", linewidth=2.0, alpha=0.9, label="Leak-Free Frontier")
    else:
        ax1.scatter(xs, ys, color="#1b9e77", s=220, facecolors='none', edgecolors="#1b9e77", linewidths=2.2, label="Global Optimum (SupCon)")
    ax1.plot([0.161, 0.161], [0.5996, 0.6227], color="#b2182b", linestyle=":", linewidth=1.5, alpha=0.7, label="Taxonomy Leakage (+0.023)")
    ax1.set_xscale("log")
    ax1.set_xlabel("Index Search Latency (ms) [Log Scale]", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Covered MAP@10 (Exact Murcko Relevance)", fontsize=10, fontweight="bold")
    ax1.set_title("Panel A: Pure Index Search Latency (Excluding Inference)", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=9, frameon=True)

    # Annotate key systems
    ax1.annotate("GIN-SupCon (Strat B) + HNSW\n(0.161 ms, MAP=0.5996)", xy=(0.161, 0.5996), xytext=(0.25, 0.55),
                 arrowprops=dict(facecolor="#1b9e77", arrowstyle="->", lw=1.2), fontsize=8, fontweight="bold")
    ax1.annotate("GIN-ProxyAnchor (Strat B)\n(0.161 ms, MAP=0.5593)", xy=(0.161, 0.5593), xytext=(0.25, 0.45),
                 arrowprops=dict(facecolor="#e63946", arrowstyle="->", lw=1.2), fontsize=8)

    # Panel B: End-to-End CPU Latency
    for s in systems:
        alpha = 0.5 if s["cat"] == "Diagnostic" else 0.9
        ax2.scatter(s["e2e"], s["map"], color=s["col"], marker=s["mark"], s=130, edgecolors="black", linewidths=0.8, alpha=alpha, zorder=5, label=s["name"])
    ax2.plot(xs_e, ys_e, color="#e63946", linestyle="--", linewidth=2.0, alpha=0.9, label="Leak-Free Frontier")
    ax2.set_xscale("log")
    ax2.set_xlabel("End-to-End Single-Query CPU Latency (ms) [Log Scale]", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Covered MAP@10 (Exact Murcko Relevance)", fontsize=10, fontweight="bold")
    ax2.set_title("Panel B: End-to-End Single-Query CPU Latency (Extraction + Search)", fontsize=11, fontweight="bold")
    ax2.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, frameon=True)

    ax2.annotate("ECFP4 + MinHash\n(0.477 ms, MAP=0.167)", xy=(0.477, 0.1668), xytext=(0.6, 0.10),
                 arrowprops=dict(facecolor="#d95f02", arrowstyle="->", lw=1.2), fontsize=8)
    ax2.annotate("ECFP4 + Exact Tanimoto\n(4.636 ms, MAP=0.229)", xy=(4.636, 0.2294), xytext=(2.0, 0.32),
                 arrowprops=dict(facecolor="#9467bd", arrowstyle="->", lw=1.2), fontsize=8)
    ax2.annotate("GIN-SupCon (Strat B) + HNSW\n(10.036 ms, MAP=0.600)", xy=(10.036, 0.5996), xytext=(12.0, 0.52),
                 arrowprops=dict(facecolor="#1b9e77", arrowstyle="->", lw=1.2), fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pareto_frontier.png"))
    plt.close()
    print("Pareto frontier figures successfully generated!")


if __name__ == "__main__":
    generate_pareto_plots()
