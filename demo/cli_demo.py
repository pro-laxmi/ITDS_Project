#!/usr/bin/env python3
"""
demo/cli_demo.py - Interactive Terminal Reproduction & Live Demonstration
CS328: Introduction to Data Science, IIT Gandhinagar
Author: Laxmidhar Panda (Roll No: 24110185)

Runs live retrieval for any test query or showcase queries directly in the terminal.
Compares learned GIN hyperspherical embeddings against classical 1024-bit ECFP4 fingerprints.
"""

import os
import sys
import time
import pickle
import argparse
import numpy as np

# Ensure project root in sys.path
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import faiss

# Paths
DATA_CACHE_PATH = os.path.join(PROJECT_DIR, "demo", "molecules_cache.pkl")
PROCESSED_CACHE_PATH = os.path.join(PROJECT_DIR, "data", "processed_cache.pkl")
CORPUS_EMB_PATH = os.path.join(PROJECT_DIR, "artifacts", "corpus_embeddings.npy")
QUERY_EMB_PATH = os.path.join(PROJECT_DIR, "artifacts", "query_embeddings.npy")

# Terminal ANSI Colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

SHOWCASE_QUERIES = [
    (528, "Drug-like Benzamide Core (Covered - GIN 8.6x Superiority)"),
    (674, "Phenyl-Piperazine Core (Covered - Pure GIN Dominance)"),
    (263, "Biphenyl Assembly (Covered - ECFP4 Zero Matches)"),
    (919, "Naphthalene Fused Core (Covered - 9/10 Matches)"),
    (1, "Unique Cycloalkyl (Uncovered - Exact Coverage Ceiling Demo)"),
]

def load_resources():
    print(f"{DIM}Loading cached datasets and vector indices...{RESET}", end="", flush=True)
    with open(DATA_CACHE_PATH, "rb") as f:
        mol_cache = pickle.load(f)

    with open(PROCESSED_CACHE_PATH, "rb") as f:
        proc_cache = pickle.load(f)

    corpus_emb = np.load(CORPUS_EMB_PATH).astype(np.float32)
    query_emb = np.load(QUERY_EMB_PATH).astype(np.float32)

    d = corpus_emb.shape[1]
    faiss_index = faiss.IndexFlatIP(d)
    faiss_index.add(corpus_emb)

    corpus_bits = proc_cache["corpus_bits"].astype(np.float32)
    query_bits = proc_cache["query_bits"].astype(np.float32)
    corpus_sums = corpus_bits.sum(axis=1)

    print(f" {GREEN}✓ Ready.{RESET}\n")
    return {
        "mol_cache": mol_cache,
        "faiss_index": faiss_index,
        "query_emb": query_emb,
        "corpus_bits": corpus_bits,
        "query_bits": query_bits,
        "corpus_sums": corpus_sums
    }

def compute_ap_recall(retrieved_scaffolds, target_scaffold, total_gt):
    if not target_scaffold or target_scaffold in ["", "__EMPTY__"]:
        return 0.0, 0.0, 0
    hits = 0
    precisions = []
    for rank, s in enumerate(retrieved_scaffolds, start=1):
        if s == target_scaffold:
            hits += 1
            precisions.append(hits / rank)
    ap = float(np.mean(precisions)) if precisions else 0.0
    rec = float(hits / total_gt) if total_gt > 0 else 0.0
    return ap, rec, hits

def run_query(q_idx, res, verbose=True):
    mol_cache = res["mol_cache"]
    q_smiles = mol_cache["query_smiles"][q_idx]
    q_scaff = mol_cache["query_scaffolds"][q_idx]
    corpus_smiles = mol_cache["corpus_smiles"]
    corpus_scaffolds = mol_cache["corpus_scaffolds"]

    is_acyclic = (q_scaff in ["", "__EMPTY__"])
    display_scaff = "Acyclic / Empty" if is_acyclic else q_scaff
    total_gt = sum(1 for s in corpus_scaffolds if s == q_scaff) if not is_acyclic else 0
    is_covered = total_gt > 0

    print("=" * 80)
    print(f"{BOLD}QUERY #{q_idx}{RESET} | Status: {GREEN if is_covered else YELLOW}{'COVERED (Ground Truth In Corpus)' if is_covered else 'UNCOVERED (0 Ground Truth in Corpus)'}{RESET}")
    print(f"  {BOLD}SMILES:{RESET}    {q_smiles}")
    print(f"  {BOLD}Scaffold:{RESET}  {display_scaff}")
    print(f"  {BOLD}Total Corpus Matches Available:{RESET} {total_gt} / {len(corpus_smiles):,} molecules")
    print("=" * 80)

    # 1. GIN Hyperspherical FAISS Retrieval
    t0 = time.perf_counter()
    q_vec = res["query_emb"][q_idx:q_idx+1]
    gin_scores, gin_indices = res["faiss_index"].search(q_vec, 10)
    gin_lat = (time.perf_counter() - t0) * 1000

    gin_scaffolds = [corpus_scaffolds[idx] for idx in gin_indices[0]]
    gin_ap, gin_rec, gin_hits = compute_ap_recall(gin_scaffolds, q_scaff, total_gt)

    # 2. ECFP4 Vectorized Tanimoto Scan
    t1 = time.perf_counter()
    q_bits = res["query_bits"][q_idx]
    inter = np.dot(res["corpus_bits"], q_bits)
    union = res["corpus_sums"] + q_bits.sum() - inter
    tanimoto = np.where(union > 0, inter / union, 0.0)
    ecfp_indices = np.argsort(-tanimoto)[:10]
    ecfp_lat = (time.perf_counter() - t1) * 1000

    ecfp_scaffolds = [corpus_scaffolds[idx] for idx in ecfp_indices]
    ecfp_ap, ecfp_rec, ecfp_hits = compute_ap_recall(ecfp_scaffolds, q_scaff, total_gt)

    # Print Top 10 Side-by-Side
    if verbose:
        print(f"\n{BOLD}{'Rank':<5} {'GIN (Hyperspherical Inner Product)':<38} {'ECFP4 (Classical Tanimoto)':<38}{RESET}")
        print("-" * 80)
        for r in range(10):
            # GIN candidate
            gid = gin_indices[0][r]
            gsc = gin_scores[0][r]
            g_match = (gin_scaffolds[r] == q_scaff) and not is_acyclic
            g_tag = f"{GREEN}[MATCH]{RESET}" if g_match else f"{RED}[DIFF ]{RESET}"
            g_str = f"#{gid:<5} Sim:{gsc:0.3f} {g_tag}"

            # ECFP candidate
            eid = ecfp_indices[r]
            esc = tanimoto[eid]
            e_match = (ecfp_scaffolds[r] == q_scaff) and not is_acyclic
            e_tag = f"{GREEN}[MATCH]{RESET}" if e_match else f"{RED}[DIFF ]{RESET}"
            e_str = f"#{eid:<5} Tan:{esc:0.3f} {e_tag}"

            print(f"{r+1:<5} {g_str:<47} {e_str}")

    # Comparative Summary Box
    print("\n" + "-" * 80)
    print(f"{BOLD}PERFORMANCE SCORECARD (Top-10 Evaluated){RESET}")
    print("-" * 80)
    print(f"  {BOLD}Metric{RESET}               {CYAN}Learned GIN (Ours){RESET}      {YELLOW}Classical ECFP4{RESET}      {BOLD}Comparison{RESET}")
    print(f"  Scaffold Hits@10     {CYAN}{gin_hits:>2} / 10{RESET}                  {YELLOW}{ecfp_hits:>2} / 10{RESET}                {'+' + str(gin_hits - ecfp_hits) if gin_hits >= ecfp_hits else str(gin_hits - ecfp_hits)}")
    print(f"  Average Prec (AP@10) {CYAN}{gin_ap:>7.4f}{RESET}                  {YELLOW}{ecfp_ap:>7.4f}{RESET}                {GREEN + f'+{((gin_ap - ecfp_ap)/ecfp_ap)*100:0.1f}%' + RESET if ecfp_ap > 0 else 'N/A'}")
    print(f"  Recall@10            {CYAN}{gin_rec:>7.4f}{RESET}                  {YELLOW}{ecfp_rec:>7.4f}{RESET}                {GREEN + f'+{((gin_rec - ecfp_rec)/ecfp_rec)*100:0.1f}%' + RESET if ecfp_rec > 0 else 'N/A'}")
    print(f"  Search Latency       {CYAN}{gin_lat:>6.2f} ms{RESET}                 {YELLOW}{ecfp_lat:>6.2f} ms{RESET}               FAISS IndexFlatIP vs Scan")
    print("-" * 80 + "\n")

    return {
        "gin_ap": gin_ap,
        "ecfp_ap": ecfp_ap,
        "gin_hits": gin_hits,
        "ecfp_hits": ecfp_hits
    }

def interactive_loop(res):
    num_queries = len(res["mol_cache"]["query_smiles"])
    print(f"{BOLD}{CYAN}--------------------------------------------------------------------------------{RESET}")
    print(f"{BOLD}  LIVE INTERACTIVE PROMPT MODE{RESET}")
    print(f"  Type any test molecule number ({BOLD}0 - {num_queries-1}{RESET}) to run live retrieval.")
    print(f"  Commands: {CYAN}'r'{RESET} = random query | {CYAN}'s'{RESET} = showcase presets | {CYAN}'q'{RESET} = quit")
    print(f"{BOLD}{CYAN}--------------------------------------------------------------------------------{RESET}\n")

    while True:
        try:
            prompt_str = f"{BOLD}Enter Molecule Number [0-{num_queries-1}] (or 'r'/'s'/'q'): {RESET}"
            user_input = input(prompt_str).strip()
            if not user_input:
                continue
            if user_input.lower() in ["q", "quit", "exit"]:
                print(f"\n{CYAN}Exiting interactive demonstration session.{RESET}\n")
                break
            elif user_input.lower() in ["r", "random"]:
                rand_idx = int(np.random.randint(0, num_queries))
                print(f"\n{DIM}>>> Random Query Selected: #{rand_idx}{RESET}")
                run_query(rand_idx, res, verbose=True)
            elif user_input.lower() in ["s", "showcase", "presets"]:
                print(f"\n{BOLD}{CYAN}Running 5 Curated Showcase Benchmark Queries...{RESET}\n")
                for q_idx, note in SHOWCASE_QUERIES:
                    print(f"\n{BOLD}{CYAN}>>> SHOWCASE: {note}{RESET}")
                    run_query(q_idx, res, verbose=True)
            else:
                try:
                    q_idx = int(user_input)
                    if 0 <= q_idx < num_queries:
                        run_query(q_idx, res, verbose=True)
                    else:
                        print(f"{RED}Error: Number out of range. Please enter an index between 0 and {num_queries-1}.{RESET}\n")
                except ValueError:
                    print(f"{RED}Unrecognized input '{user_input}'. Enter an integer 0-{num_queries-1}, 'r' for random, or 'q' to quit.{RESET}\n")
        except (KeyboardInterrupt, EOFError):
            print(f"\n\n{CYAN}Session terminated.{RESET}\n")
            break

def main():
    parser = argparse.ArgumentParser(description="CS328 ITDS: Interactive Molecular Retrieval CLI")
    parser.add_argument("--query_idx", type=int, default=None, help="Specific query index to test (0-999). If omitted, starts interactive live input mode.")
    parser.add_argument("--all_showcase", action="store_true", help="Run all 5 curated showcase queries sequentially.")
    parser.add_argument("--interactive", "-i", action="store_true", help="Force interactive prompt loop.")
    args = parser.parse_args()

    print(f"\n{BOLD}================================================================================{RESET}")
    print(f"{BOLD}{CYAN}  MOLECULAR SIMILARITY SEARCH USING GNNs & APPROXIMATE NEAREST NEIGHBORS{RESET}")
    print(f"  CS328: Introduction to Data Science | IIT Gandhinagar")
    print(f"  Author: Laxmidhar Panda (24110185)")
    print(f"{BOLD}================================================================================{RESET}\n")

    res = load_resources()

    if args.all_showcase:
        print(f"{BOLD}Running Full Showcase Battery (5 Illustrative Benchmark Queries)...{RESET}\n")
        for q_idx, note in SHOWCASE_QUERIES:
            print(f"{BOLD}{CYAN}>>> SHOWCASE QUERY: {note}{RESET}")
            run_query(q_idx, res, verbose=True)
            time.sleep(0.5)
    elif args.query_idx is not None and not args.interactive:
        run_query(args.query_idx % len(res["mol_cache"]["query_smiles"]), res, verbose=True)
    else:
        interactive_loop(res)

if __name__ == "__main__":
    main()
