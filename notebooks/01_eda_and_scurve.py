"""
01_eda_and_scurve.py - Verification of MinHash Banding S-Curve Phase Transition
CS328: Introduction to Data Science
Author: Laxmidhar Panda (24110185)

Integrity Check 1:
Plot theoretical Candidate Probability S-curve P = 1 - (1 - J^r)^b and overlay
empirical candidate hit rates from set_hashing.py to ensure banding logic is mathematically sound.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Set, Tuple

# Ensure src is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.set_hashing import MinHasher, LSHBandingIndex


def generate_synthetic_shingle_pairs(
    num_pairs: int = 4000,
    universe_size: int = 10000,
    seed: int = 42
) -> List[Tuple[Set[int], Set[int], float]]:
    """
    Generate synthetic shingle set pairs spanning the full [0, 1] Jaccard similarity range.
    Each pair (A, B) has known exact Jaccard similarity J(A, B) = |A n B| / |A u B|.
    """
    rng = np.random.default_rng(seed)
    pairs = []

    # Sample target similarities uniformly across [0.0, 1.0]
    target_sims = rng.uniform(0.0, 1.0, size=num_pairs)

    for target_j in target_sims:
        # Base set size between 30 and 100 elements (resembling ECFP4 shingle counts)
        size_a = rng.integers(30, 80)
        # Solve for intersection and union:
        # J = I / U = I / (size_a + size_b - I)
        # Let size_b approx size_a: J = I / (2*size_a - I) => I = 2*size_a*J / (1 + J)
        intersect_size = int(round((2 * size_a * target_j) / (1.0 + target_j)))
        intersect_size = max(0, min(size_a, intersect_size))

        common_elements = set(rng.choice(universe_size, size=intersect_size, replace=False))
        remaining_pool = list(set(range(universe_size)) - common_elements)

        diff_a_size = size_a - intersect_size
        diff_b_size = max(0, int(round((size_a - intersect_size) * rng.uniform(0.8, 1.2))))

        diff_a = set(rng.choice(remaining_pool, size=diff_a_size, replace=False))
        remaining_pool = list(set(remaining_pool) - diff_a)
        diff_b = set(rng.choice(remaining_pool, size=diff_b_size, replace=False))

        set_a = common_elements | diff_a
        set_b = common_elements | diff_b

        exact_j = len(set_a & set_b) / len(set_a | set_b) if (set_a or set_b) else 1.0
        pairs.append((set_a, set_b, exact_j))

    return pairs


def verify_scurve(
    b: int = 32,
    r: int = 4,
    num_pairs: int = 4000,
    output_path: str = "artifacts/s_curve_verification.png"
):
    print(f"=== Running Integrity Check 1: S-Curve Verification (b={b}, r={r}, k={b*r}) ===")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    hasher = MinHasher(num_hashes=b * r, seed=42)
    lsh = LSHBandingIndex(b=b, r=r)
    theoretical_thresh = (1.0 / b) ** (1.0 / r)
    print(f"Theoretical 50% collision threshold t* = {theoretical_thresh:.4f}")

    print(f"Generating {num_pairs} evaluation set pairs spanning J in [0, 1]...")
    pairs = generate_synthetic_shingle_pairs(num_pairs=num_pairs, seed=42)

    true_jaccards = []
    collisions = []

    for set_a, set_b, exact_j in pairs:
        sig_a = hasher.compute_signature(set_a)
        sig_b = hasher.compute_signature(set_b)

        # Check if they collide in at least one band
        collide = False
        for band_idx in range(b):
            start = band_idx * r
            end = start + r
            if np.array_equal(sig_a[start:end], sig_b[start:end]):
                collide = True
                break

        true_jaccards.append(exact_j)
        collisions.append(1 if collide else 0)

    true_jaccards = np.array(true_jaccards)
    collisions = np.array(collisions)

    # Bin empirical data to plot probabilities
    bins = np.linspace(0.0, 1.0, 21)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    empirical_probs = []
    sample_counts = []

    for i in range(len(bins) - 1):
        mask = (true_jaccards >= bins[i]) & (true_jaccards < bins[i+1])
        if np.sum(mask) > 0:
            prob = np.mean(collisions[mask])
            empirical_probs.append(prob)
            sample_counts.append(np.sum(mask))
        else:
            empirical_probs.append(np.nan)
            sample_counts.append(0)

    bin_centers = np.array(bin_centers)
    empirical_probs = np.array(empirical_probs)

    # Theoretical curve
    s_fine = np.linspace(0.0, 1.0, 200)
    p_theoretical = lsh.theoretical_collision_prob(s_fine, b=b, r=r)

    # Calculate Mean Absolute Error (MAE) on valid bins
    valid_mask = ~np.isnan(empirical_probs)
    p_theo_at_bins = lsh.theoretical_collision_prob(bin_centers[valid_mask], b=b, r=r)
    mae = float(np.mean(np.abs(empirical_probs[valid_mask] - p_theo_at_bins)))
    max_err = float(np.max(np.abs(empirical_probs[valid_mask] - p_theo_at_bins)))

    print(f"Empirical vs. Theoretical Fit: MAE = {mae:.4f}, Max Absolute Error = {max_err:.4f}")
    assert mae < 0.05, f"Integrity Check Failed: MAE {mae:.4f} exceeds 0.05 threshold!"
    print(">> INTEGRITY CHECK 1 PASSED: Empirical banding hit rate faithfully tracks theoretical S-curve.")

    # Create publication-quality plot
    plt.figure(figsize=(9, 6), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    plt.plot(s_fine, p_theoretical, label=f"Theoretical $P = 1 - (1 - s^{r})^{{{b}}}$", color="#1f77b4", linewidth=2.5)
    plt.plot(bin_centers[valid_mask], empirical_probs[valid_mask], "ro-", markersize=6, linewidth=1.5,
             label=f"Empirical Hit Rate (N={num_pairs}, MAE={mae:.3f})")
    plt.axvline(theoretical_thresh, color="black", linestyle="--", alpha=0.7,
                label=f"Phase Transition $t^* \\approx {theoretical_thresh:.3f}$")
    plt.axhline(0.5, color="gray", linestyle=":", alpha=0.5)

    plt.title(f"LSH MinHash Banding Phase Transition (b={b} bands, r={r} rows, k={b*r})", fontsize=14, fontweight="bold")
    plt.xlabel("True Jaccard Similarity $J(A, B)$", fontsize=12)
    plt.ylabel("Candidate Collision Probability $P$", fontsize=12)
    plt.xlim(0.0, 1.0)
    plt.ylim(-0.02, 1.02)
    plt.legend(loc="lower right", fontsize=11, frameon=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved verification plot to: {output_path}")


if __name__ == "__main__":
    verify_scurve(b=32, r=4, num_pairs=4000)
