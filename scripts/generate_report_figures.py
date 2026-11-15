#!/usr/bin/env python3
"""
scripts/generate_report_figures.py
Generates publication-grade, de-cluttered figures with large, human-readable text:
  0. fig0_tradeoff.png
  1. fig1_system_workflow.png (De-cluttered 3-stage linear pipeline)
  2. fig2_gin_architecture.png (Enlarged formulas and clear component dataflow)
  3. fig3_molecular_graph.png (Enlarged card text and chemical structure)
  4. fig4_performance_comparison.png
  5. fig5_evaluation_coverage.png
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image

FIGURES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['mathtext.fontset'] = 'cm'

def generate_fig0_tradeoff():
    """Generates Figure 1: Graphical representation of the molecular retrieval tradeoff."""
    fig, ax = plt.subplots(figsize=(10.6, 5.4), dpi=300)
    ax.set_xlim(0, 10.6)
    ax.set_ylim(0, 5.4)
    ax.axis('off')

    frame = FancyBboxPatch((0.15, 0.15), 10.3, 5.1, boxstyle="round,pad=0.06,rounding_size=0.12",
                           facecolor='#ffffff', edgecolor='#cbd5e1', linewidth=1.5)
    ax.add_patch(frame)

    ax.text(5.3, 4.95, "Molecular Similarity Retrieval: Chemical Fidelity vs. Search Latency",
            ha='center', va='center', fontsize=13.5, fontweight='bold', color='#0b1d3a')
    ax.text(5.3, 4.62, "Comparison of Combinatorial, Discrete, and Geometric Deep Learning Approaches",
            ha='center', va='center', fontsize=9.8, color='#475569')

    # X-axis: Latency
    ax.annotate('', xy=(9.8, 0.95), xytext=(1.1, 0.95),
                arrowprops=dict(arrowstyle="->,head_width=0.38,head_length=0.48", color='#334155', lw=2.2))
    ax.text(1.1, 0.55, "High Latency / Combinatorial\n(Exponential Worst-Case)", ha='left', va='center',
            fontsize=8.8, fontweight='bold', color='#64748b')
    ax.text(9.8, 0.55, "Fast Retrieval / Sub-linear\n(Vector / Hash Scanning)", ha='right', va='center',
            fontsize=8.8, fontweight='bold', color='#047857')
    ax.text(5.3, 0.62, "RETRIEVAL LATENCY & COMPLEXITY", ha='center', va='center',
            fontsize=9.8, fontweight='bold', color='#1e293b')

    # Y-axis: Chemical Fidelity
    ax.annotate('', xy=(1.1, 4.2), xytext=(1.1, 0.95),
                arrowprops=dict(arrowstyle="->,head_width=0.38,head_length=0.48", color='#334155', lw=2.2))
    ax.text(0.85, 4.05, "High\nFidelity", ha='right', va='center', fontsize=8.8, fontweight='bold', color='#1e3a8a')
    ax.text(0.85, 1.25, "Low\nFidelity", ha='right', va='center', fontsize=8.8, fontweight='bold', color='#b91c1c')

    # Box 1: Exact Subgraph Isomorphism
    box1 = FancyBboxPatch((1.6, 2.75), 3.0, 1.45, boxstyle="round,pad=0.06,rounding_size=0.10",
                          facecolor='#fee2e2', edgecolor='#dc2626', linewidth=1.8)
    ax.add_patch(box1)
    ax.text(3.1, 3.90, "Exact Subgraph Isomorphism", ha='center', va='center', fontsize=10.5, fontweight='bold', color='#991b1b')
    ax.text(3.1, 3.52, "• NP-Complete (Ullmann, 1976)", ha='center', va='center', fontsize=8.8, color='#7f1d1d')
    ax.text(3.1, 3.22, "• Exponential worst-case: O(|V|^k)", ha='center', va='center', fontsize=8.8, color='#7f1d1d')
    ax.text(3.1, 2.92, "• Intractable for large libraries", ha='center', va='center', fontsize=8.8, color='#7f1d1d')

    # Box 2: Discrete Fingerprints
    box2 = FancyBboxPatch((6.4, 1.35), 3.6, 1.45, boxstyle="round,pad=0.06,rounding_size=0.10",
                          facecolor='#fef3c7', edgecolor='#d97706', linewidth=1.8)
    ax.add_patch(box2)
    ax.text(8.2, 2.50, "Discrete Fingerprints (ECFP4)", ha='center', va='center', fontsize=10.5, fontweight='bold', color='#92400e')
    ax.text(8.2, 2.12, "• Fast linear bitwise scan / hash indexing", ha='center', va='center', fontsize=8.8, color='#78350f')
    ax.text(8.2, 1.82, "• Lossy bit folding & rigid hash collisions", ha='center', va='center', fontsize=8.8, color='#78350f')
    ax.text(8.2, 1.52, "• Cannot learn continuous graph topology", ha='center', va='center', fontsize=8.8, color='#78350f')

    # Box 3: Geometric Deep Learning (Target)
    box3 = FancyBboxPatch((4.9, 2.75), 5.3, 1.55, boxstyle="round,pad=0.06,rounding_size=0.12",
                          facecolor='#dcfce7', edgecolor='#16a34a', linewidth=2.0)
    ax.add_patch(box3)
    ax.text(7.55, 4.02, "Learned GIN Hyperspherical Embeddings (This Work)", ha='center', va='center', fontsize=11.0, fontweight='bold', color='#14532d')
    ax.text(7.55, 3.62, "• Polynomial encoder pass: O(|V| + |E|) via 4-layer GIN", ha='center', va='center', fontsize=9.0, color='#166534')
    ax.text(7.55, 3.30, "• Fast vector retrieval via FAISS inner product ranking", ha='center', va='center', fontsize=9.0, color='#166534')
    ax.text(7.55, 2.98, "• Metric learning captures scaffold topology & linker chemistry", ha='center', va='center', fontsize=9.0, color='#166534')

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig0_tradeoff.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated {out_path}")

def generate_fig1_system_workflow():
    """Generates Figure 4: End-to-end retrieval system workflow (Spacious, long vertical/horizontal arrows, breathable cards)."""
    fig, ax = plt.subplots(figsize=(11.4, 7.4), dpi=300)
    ax.set_xlim(0, 11.4)
    ax.set_ylim(0, 7.4)
    ax.axis('off')

    c_blue = '#eff6ff'; c_blue_b = '#2563eb'
    c_amber = '#fef3c7'; c_amber_b = '#d97706'
    c_green = '#f0fdf4'; c_green_b = '#16a34a'
    c_purple = '#faf5ff'; c_purple_b = '#7c3aed'
    c_rose = '#fff1f2'; c_rose_b = '#e11d48'
    c_gray = '#f8fafc'; c_gray_b = '#475569'

    def draw_card(x, y, w, h, title, lines, bg_col, border_col, title_col=None, pad_x=0.15, title_fs=11.5, text_fs=9.3):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
                             facecolor=bg_col, edgecolor=border_col, linewidth=1.8, zorder=2)
        ax.add_patch(box)
        t_col = title_col if title_col else border_col
        ax.text(x + w/2, y + h - 0.28, title, ha='center', va='center',
                fontsize=title_fs, fontweight='bold', color=t_col, zorder=3)
        n = len(lines)
        if n > 0:
            start_y = y + h - 0.58
            gap = (h - 0.72) / max(n - 1, 1) if n > 1 else 0
            for i, line in enumerate(lines):
                cur_y = start_y - i * gap if n > 1 else y + (h - 0.35)/2
                ax.text(x + pad_x, cur_y, line, ha='left', va='center',
                        fontsize=text_fs, color='#0f172a', zorder=3)

    def draw_arrow(x1, y1, x2, y2, label="", ha='center', va='bottom', off_x=0, off_y=0.10, fs=9.2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->,head_width=0.40,head_length=0.50",
                                    color='#1e293b', lw=2.2), zorder=4)
        if label:
            mx, my = (x1 + x2)/2 + off_x, (y1 + y2)/2 + off_y
            ax.text(mx, my, label, ha=ha, va=va,
                    fontsize=fs, fontweight='bold', color='#1e293b', zorder=5)

    # Title Banner with generous breathing room
    ax.text(5.7, 7.12, "End-to-End Molecular Graph Retrieval Pipeline", ha='center', va='center',
            fontsize=15.5, fontweight='bold', color='#0b1d3a')
    ax.text(5.7, 6.75, "Dual Architecture: Learned GIN Hyperspherical Embeddings vs. ECFP4 Fingerprints",
            ha='center', va='center', fontsize=10.5, color='#475569')

    # STAGE 1: Data Preparation & Partitioning (y = 5.05 to 6.45, h = 1.40)
    draw_card(0.4, 5.05, 2.9, 1.40, "1. Raw Data & Graph Encoding",
              ["• 12,000 ZINC SMILES molecules",
               "• Nodes: 28 Atom Tokens, Chirality",
               "• Edges: 3 Active Bond Types (1, 2, 3)"], c_blue, c_blue_b, text_fs=9.2)

    # Longer horizontal arrow between 1 and 2
    draw_arrow(3.3, 5.75, 4.3, 5.75, "Partition", fs=9.2)

    draw_card(4.3, 5.05, 3.3, 1.40, "2. Corpus & Query Split",
              ["• Candidate Corpus: 11,000 Molecules",
               "  - Train: 10,000 | Validation: 1,000",
               "• Test Queries: 1,000 Molecules (275 Cov)"], c_amber, c_amber_b, text_fs=9.2)

    # Longer horizontal arrow between 2 and 3
    draw_arrow(7.6, 5.75, 8.5, 5.75, "10k Train", fs=9.2)

    draw_card(8.5, 5.05, 2.5, 1.40, "3. Strategy B Taxonomy",
              ["• Train-only Bemis-Murcko",
               "• 241 Scaffolds + 3 Acyclic = 244 Cls",
               "• 1,894 Labeled | 8,106 Excluded"], c_amber, c_amber_b, text_fs=9.2)

    # STAGE 2: Dual Search Branches (y = 2.65 to 4.35, h = 1.70)
    # Long vertical arrows from Stage 1 down to Stage 2 (length = 0.70 units!)
    draw_arrow(1.85, 5.05, 1.85, 4.35, "SMILES", ha='left', off_x=0.12, off_y=0, fs=9.2)
    draw_card(0.4, 2.65, 2.9, 1.70, "Classical ECFP4 Baseline",
              ["• 1024-bit Morgan (Radius 2)",
               "• Exhaustive Linear Bitwise Scan",
               "• Exact Tanimoto (Jaccard) Scoring",
               "• Rigid bit collisions & folding",
               "• Static representation (no learning)"], c_gray, c_gray_b, text_fs=9.1)

    # Right: Learned GIN Branch with long vertical arrows
    draw_arrow(5.95, 5.05, 5.95, 4.35, "Corpus & Queries", ha='right', off_x=-0.12, off_y=0, fs=9.2)
    draw_arrow(9.75, 5.05, 9.75, 4.35, "Supervision", ha='left', off_x=0.12, off_y=0, fs=9.2)
    draw_card(3.7, 2.65, 7.3, 1.70, "Learned GIN Hyperspherical Retrieval Branch (This Work)",
              ["• 4-Layer GIN Encoder with GINEConv (incorporating directional bond embeddings)",
               "• Dual Global Graph Pooling (Sum Pool + Mean Pool = 128-D) -> 2-Layer Projection Head",
               "• L2 Hyperspherical Normalization: All embeddings constrained to unit hypersphere S^255",
               "• Corpus Index: 11,000 vectors indexed in FAISS via IndexFlatIP (Exact Inner Product)",
               "• Query Retrieval: Cosine rank-equivalent to Euclidean distance on S^255 -> Top-10 hits"],
              c_green, c_green_b, text_fs=9.2)

    # STAGE 3: Exact Scaffold Benchmark Evaluation (y = 0.75 to 1.95, h = 1.20)
    # Long vertical arrows from Stage 2 down to Stage 3 (length = 0.70 units!)
    draw_arrow(1.85, 2.65, 1.85, 1.95, "Top-10 Hits", ha='left', off_x=0.12, off_y=0, fs=9.0)
    draw_arrow(7.35, 2.65, 7.35, 1.95, "Top-10 Hits", ha='left', off_x=0.12, off_y=0, fs=9.0)

    draw_card(0.4, 0.75, 3.3, 1.20, "Exact Scaffold Ground Truth",
              ["• Stored keys from exact_scaffolds.pkl",
               "• Relevance: exact canonical BM equality",
               "• Excludes empty acyclic pseudo-classes",
               "• Independent of GNN feature spaces"], c_purple, c_purple_b, text_fs=8.9)

    draw_card(4.1, 0.75, 3.4, 1.20, "Covered Test Queries (N = 275)",
              ["• GIN-SupCon:     0.5996 MAP | 0.5939 Rec",
               "• ProxyAnchor:   0.5593 MAP | 0.5244 Rec",
               "• ECFP4 Tanimoto: 0.2294 MAP | 0.3116 Rec",
               "• GIN Relative Gain: +161.4% MAP@10"], c_rose, c_rose_b, text_fs=8.9)

    draw_card(7.9, 0.75, 3.1, 1.20, "Full Test Set (N = 1,000)",
              ["• GIN-SupCon:     0.1649 MAP | 0.1633 Rec",
               "• ProxyAnchor:   0.1538 MAP | 0.1442 Rec",
               "• ECFP4 Tanimoto: 0.0631 MAP | 0.0857 Rec",
               "• Bounded by 27.5% coverage bound"], c_rose, c_rose_b, text_fs=8.9)

    # Bottom Summary Strip with comfortable padding
    banner = FancyBboxPatch((0.4, 0.10), 10.6, 0.44, boxstyle="round,pad=0.04,rounding_size=0.08",
                            facecolor='#ffffff', edgecolor='#94a3b8', linewidth=1.2, linestyle='--')
    ax.add_patch(banner)
    ax.text(5.7, 0.33, "Summary: Dual-branch evaluation rigorously evaluates GIN representations against classical 1024-bit ECFP4 fingerprints,",
            ha='center', va='center', fontsize=9.2, color='#1e293b')
    ax.text(5.7, 0.19, "relying strictly on stored Bemis-Murcko scaffold keys for ground truth relevance across 11,000 candidate corpus molecules.",
            ha='center', va='center', fontsize=9.2, color='#1e293b', fontweight='bold')

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig1_system_workflow.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated {out_path}")

def generate_fig2_gin_architecture():
    """Generates Figure 5: Detailed GIN Encoder architecture (Spacious, prominent equations, long arrows, large readable text)."""
    fig, ax = plt.subplots(figsize=(12.0, 7.4), dpi=300)
    ax.set_xlim(0, 12.0)
    ax.set_ylim(0, 7.4)
    ax.axis('off')

    c_blue = '#eff6ff'; c_blue_b = '#2563eb'
    c_amber = '#fef3c7'; c_amber_b = '#d97706'
    c_green = '#f0fdf4'; c_green_b = '#16a34a'
    c_mlp = '#fff7ed'; c_mlp_b = '#ea580c'
    c_purple = '#faf5ff'; c_purple_b = '#7c3aed'
    c_rose = '#fff1f2'; c_rose_b = '#e11d48'

    def draw_card(x, y, w, h, title, lines, bg_col, border_col, title_col=None, pad_x=0.14, title_fs=11.5, text_fs=9.2):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
                             facecolor=bg_col, edgecolor=border_col, linewidth=1.8, zorder=2)
        ax.add_patch(box)
        t_col = title_col if title_col else border_col
        ax.text(x + w/2, y + h - 0.28, title, ha='center', va='center',
                fontsize=title_fs, fontweight='bold', color=t_col, zorder=3)
        n = len(lines)
        if n > 0:
            start_y = y + h - 0.58
            gap = (h - 0.72) / max(n - 1, 1) if n > 1 else 0
            for i, line in enumerate(lines):
                cur_y = start_y - i * gap if n > 1 else y + (h - 0.35)/2
                ax.text(x + pad_x, cur_y, line, ha='left', va='center',
                        fontsize=text_fs, color='#0f172a', zorder=3)

    def draw_arrow(x1, y1, x2, y2, label="", ha='center', va='bottom', off_x=0, off_y=0.10, fs=9.5):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->,head_width=0.40,head_length=0.50",
                                    color='#1e293b', lw=2.2), zorder=4)
        if label:
            mx, my = (x1 + x2)/2 + off_x, (y1 + y2)/2 + off_y
            ax.text(mx, my, label, ha=ha, va=va,
                    fontsize=fs, fontweight='bold', color='#1e293b', zorder=5)

    # Title Banner with generous breathing room
    ax.text(6.0, 7.12, "Graph Isomorphism Network (GIN) Encoder Architecture", ha='center', va='center',
            fontsize=15.5, fontweight='bold', color='#0b1d3a')
    ax.text(6.0, 6.75, r"Layer-by-layer dataflow from discrete molecular tokens to unit hyperspherical embeddings $\mathbb{S}^{255}$",
            ha='center', va='center', fontsize=10.5, color='#475569')

    # STEP 1: Input Categorical Tokens (x = 0.40 to 2.35, w = 1.95)
    # Upper card: Atom tokens (y = 3.95 to 5.85, h = 1.90)
    draw_card(0.40, 3.95, 1.95, 1.90, "Atom Token Lookup",
              ["• 28 ZINC Atom Tokens",
               "• Embedding(28, 64)",
               r"• $h_v^{(0)} = \mathbf{W}_{atom}(x_v)$",
               r"• Initial $h_v^{(0)} \in \mathbb{R}^{64}$"], c_blue, c_blue_b, text_fs=9.2)

    # Lower card: Bond tokens (y = 1.85 to 3.75, h = 1.90)
    draw_card(0.40, 1.85, 1.95, 1.90, "Bond Token Lookup",
              ["• 3 Active Types (1, 2, 3)",
               "• Embedding(5, 64)",
               "  (0 null, 4 reserved)",
               r"• $e_{uv} = \mathbf{W}_{bond}(e_{uv})$",
               r"• Bond vector $e_{uv} \in \mathbb{R}^{64}$"], c_amber, c_amber_b, text_fs=9.2)

    # Long horizontal/diagonal arrows from Tokens into GIN Message Passing (Length ~0.90 units!)
    draw_arrow(2.35, 4.90, 3.25, 4.90, fs=9.2)
    draw_arrow(2.35, 2.80, 3.25, 3.35, fs=9.2)

    # STEP 2: 4 x GIN Message Passing Layers (x = 3.25 to 6.55, w = 3.30, y = 1.85, h = 4.00)
    gine_box = FancyBboxPatch((3.25, 1.85), 3.30, 4.00, boxstyle="round,pad=0.06,rounding_size=0.14",
                              facecolor=c_green, edgecolor=c_green_b, linewidth=2.0, zorder=2)
    ax.add_patch(gine_box)
    ax.text(4.90, 5.52, r"4 $\times$ GIN Message Passing Layers", ha='center', va='center',
            fontsize=12.0, fontweight='bold', color='#14532d', zorder=3)

    # Message Equation
    ax.text(4.90, 5.02, r"$m_{uv}^{(l)} = \mathrm{ReLU}\left(h_u^{(l-1)} + \mathbf{W}_e^{(l)} e_{uv}\right)$",
            ha='center', va='center', fontsize=11.2, color='#0f172a', zorder=3)
    ax.text(4.90, 4.68, "Directional Edge Message Integration", ha='center', va='center',
            fontsize=9.0, color='#15803d', fontstyle='italic', zorder=3)

    # Update Equation
    ax.text(4.90, 4.15, r"$h_v^{(l)} = \mathrm{MLP}^{(l)}\left( (1+\epsilon^{(l)}) h_v^{(l-1)} + \sum_{u \in \mathcal{N}(v)} m_{uv}^{(l)} \right)$",
            ha='center', va='center', fontsize=10.8, color='#0f172a', zorder=3)
    ax.text(4.90, 3.80, "Injective Multiset Aggregation & Node Update", ha='center', va='center',
            fontsize=9.0, color='#15803d', fontstyle='italic', zorder=3)

    # Internal MLP Panel inside GIN block
    mlp_box = FancyBboxPatch((3.45, 2.00), 2.90, 1.55, boxstyle="round,pad=0.04,rounding_size=0.10",
                             facecolor=c_mlp, edgecolor=c_mlp_b, linewidth=1.5, zorder=3)
    ax.add_patch(mlp_box)
    ax.text(4.90, 3.25, r"Internal 2-Layer Node MLP", ha='center', va='center',
            fontsize=10.2, fontweight='bold', color='#c2410c', zorder=4)
    ax.text(4.90, 2.88, r"$\mathrm{Linear}(64 \to 64) \to \mathrm{BatchNorm} \to \mathrm{GELU}$",
            ha='center', va='center', fontsize=8.8, color='#0f172a', zorder=4)
    ax.text(4.90, 2.52, r"$\mathrm{Linear}(64 \to 64) \to \mathrm{BatchNorm} \to \mathrm{GELU}$",
            ha='center', va='center', fontsize=8.8, color='#0f172a', zorder=4)
    ax.text(4.90, 2.18, r"Learnable scalar $\epsilon^{(l)}$, hidden dimension = 64",
            ha='center', va='center', fontsize=8.5, color='#9a3412', fontstyle='italic', zorder=4)

    # Long arrow from GIN to Global Pooling (Length = 0.95 units!)
    draw_arrow(6.55, 3.85, 7.50, 3.85, r"$h_v^{(4)}$", ha='center', off_y=0.15, fs=10.0)

    # STEP 3: Dual Global Graph Pooling (x = 7.50 to 9.15, w = 1.65, y = 1.85, h = 4.00)
    pool_box = FancyBboxPatch((7.50, 1.85), 1.65, 4.00, boxstyle="round,pad=0.06,rounding_size=0.12",
                              facecolor=c_purple, edgecolor=c_purple_b, linewidth=1.8, zorder=2)
    ax.add_patch(pool_box)
    ax.text(8.32, 5.52, "Dual Global Pooling", ha='center', va='center',
            fontsize=10.8, fontweight='bold', color=c_purple_b, zorder=3)

    ax.text(8.32, 4.95, "Sum Pooling", ha='center', va='center', fontsize=10.0, fontweight='bold', color='#4c1d95', zorder=3)
    ax.text(8.32, 4.60, r"$\sum_{v \in V} h_v^{(4)} \in \mathbb{R}^{64}$", ha='center', va='center', fontsize=9.8, color='#0f172a', zorder=3)
    ax.text(8.32, 4.25, "(Captures graph size)", ha='center', va='center', fontsize=8.5, color='#6b21a8', fontstyle='italic', zorder=3)

    ax.text(8.32, 3.65, "Mean Pooling", ha='center', va='center', fontsize=10.0, fontweight='bold', color='#4c1d95', zorder=3)
    ax.text(8.32, 3.30, r"$\frac{1}{|V|}\sum h_v^{(4)} \in \mathbb{R}^{64}$", ha='center', va='center', fontsize=9.8, color='#0f172a', zorder=3)
    ax.text(8.32, 2.95, "(Captures atom density)", ha='center', va='center', fontsize=8.5, color='#6b21a8', fontstyle='italic', zorder=3)

    ax.text(8.32, 2.40, "Concatenation", ha='center', va='center', fontsize=10.0, fontweight='bold', color='#4c1d95', zorder=3)
    ax.text(8.32, 2.08, r"$h_G \in \mathbb{R}^{128}$", ha='center', va='center', fontsize=10.2, fontweight='bold', color='#0f172a', zorder=3)

    # Long arrow from Pooling to Projection Head (Length = 0.95 units!)
    draw_arrow(9.15, 3.85, 10.10, 3.85, r"$h_G$", ha='center', off_y=0.15, fs=10.0)

    # STEP 4: Projection Head & L2 Normalization (x = 10.10 to 11.60, w = 1.50, y = 1.85, h = 4.00)
    proj_box = FancyBboxPatch((10.10, 1.85), 1.50, 4.00, boxstyle="round,pad=0.06,rounding_size=0.12",
                              facecolor=c_rose, edgecolor=c_rose_b, linewidth=1.8, zorder=2)
    ax.add_patch(proj_box)
    ax.text(10.85, 5.52, "Projection &\nNormalization", ha='center', va='center',
            fontsize=10.2, fontweight='bold', color=c_rose_b, zorder=3)

    ax.text(10.85, 4.80, "Projection Head", ha='center', va='center', fontsize=9.8, fontweight='bold', color='#9f1239', zorder=3)
    ax.text(10.85, 4.45, r"$\mathrm{Linear}(128 \to 256)$", ha='center', va='center', fontsize=8.2, color='#0f172a', zorder=3)
    ax.text(10.85, 4.15, r"$\mathrm{LayerNorm}(256)$", ha='center', va='center', fontsize=8.2, color='#0f172a', zorder=3)
    ax.text(10.85, 3.85, r"$\mathrm{GELU}$", ha='center', va='center', fontsize=8.5, color='#0f172a', zorder=3)
    ax.text(10.85, 3.55, r"$\mathrm{Linear}(256 \to 256)$", ha='center', va='center', fontsize=8.2, color='#0f172a', zorder=3)

    ax.text(10.85, 2.95, r"$L_2$ Normalization", ha='center', va='center', fontsize=9.8, fontweight='bold', color='#9f1239', zorder=3)
    ax.text(10.85, 2.58, r"$z = \frac{\tilde{z}_G}{\|\tilde{z}_G\|_2}$", ha='center', va='center', fontsize=10.8, color='#0f172a', zorder=3)
    ax.text(10.85, 2.15, r"$z \in \mathbb{S}^{255}, \|z\|_2 = 1$", ha='center', va='center', fontsize=9.2, fontweight='bold', color='#be123c', zorder=3)

    # STEP 5: Bottom Retrieval Panel (x = 0.40 to 11.60, w = 11.20, y = 0.15 to 1.45, h = 1.30)
    ret_box = FancyBboxPatch((0.40, 0.15), 11.20, 1.30, boxstyle="round,pad=0.06,rounding_size=0.12",
                             facecolor='#f8fafc', edgecolor='#64748b', linewidth=1.5, zorder=2)
    ax.add_patch(ret_box)
    ax.text(6.0, 1.22, "Downstream Retrieval via Hyperspherical Inner Product Ranking (FAISS IndexFlatIP)",
            ha='center', va='center', fontsize=11.5, fontweight='bold', color='#0f172a', zorder=3)
    ax.text(6.0, 0.88, r"Because all embeddings $q, c \in \mathbb{S}^{255}$ are unit-normalized ($\|q\|_2 = \|c\|_2 = 1$):   $\mathrm{Similarity}(q, c) = \langle q, c \rangle = q^\top c = \cos \theta_{q, c}$",
            ha='center', va='center', fontsize=10.0, color='#1e293b', zorder=3)
    ax.text(6.0, 0.54, r"Euclidean Distance Isomorphism:   $\|q - c\|_2^2 = \|q\|_2^2 + \|c\|_2^2 - 2\langle q, c \rangle = 2 - 2\langle q, c \rangle$",
            ha='center', va='center', fontsize=10.2, fontweight='bold', color='#0369a1', zorder=3)
    ax.text(6.0, 0.26, "Inner product ranking is strictly monotonic with Euclidean nearest neighbor distance, enabling exact sub-linear vector retrieval.",
            ha='center', va='center', fontsize=9.2, color='#475569', zorder=3)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig2_gin_architecture.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated {out_path}")

def generate_fig3_molecular_graph():
    """Generates Figure 2: Real molecule (ZINC Index 2) with enlarged text and balanced layout."""
    smiles = "CN(CC1=CC=C(Cl)C=C1)C(=O)NC1CC1"
    mol = Chem.MolFromSmiles(smiles)
    
    drawer = rdMolDraw2D.MolDraw2DCairo(1000, 800)
    opts = drawer.drawOptions()
    opts.addAtomIndices = True
    opts.bondLineWidth = 3.5
    opts.padding = 0.08
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    temp_img_path = os.path.join(FIGURES_DIR, "temp_mol_2d.png")
    with open(temp_img_path, "wb") as f:
        f.write(drawer.GetDrawingText())

    fig = plt.figure(figsize=(10.6, 5.6), dpi=300)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.4], wspace=0.10)

    ax1 = fig.add_subplot(gs[0])
    img = Image.open(temp_img_path)
    ax1.imshow(img)
    ax1.axis('off')
    ax1.set_title("(a) 2D Chemical Structure (ZINC Train Index 2)\nSMILES: CN(CC1=CC=C(Cl)C=C1)C(=O)NC1CC1\n(16 Heavy Atoms, 17 Covalent Bonds)",
                  fontsize=10.5, fontweight='bold', color='#0b1d3a', pad=10)

    if os.path.exists(temp_img_path):
        os.remove(temp_img_path)

    ax2 = fig.add_subplot(gs[1])
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title("(b) GNN Attributed Graph Features & Scaffold Extraction",
                  fontsize=11.0, fontweight='bold', color='#0b1d3a', pad=10)

    card = FancyBboxPatch((0.15, 0.15), 9.7, 9.6, boxstyle="round,pad=0.08,rounding_size=0.12",
                          facecolor='#f8fafc', edgecolor='#94a3b8', linewidth=1.5)
    ax2.add_patch(card)

    ax2.text(0.5, 9.2, "Atom (Node) Attributes: [Atomic Number, Chirality]",
             fontsize=10.5, fontweight='bold', color='#1e3a8a')
    
    atom_lines = [
        "• Node 0: C (Z=6, Chiral=0)  [N-Methyl]",
        "• Node 1: N (Z=7, Chiral=0)  [Amine Linker]",
        "• Node 2: C (Z=6, Chiral=0)  [Benzyl CH2]",
        "• Nodes 3–8: Chlorobenzene Core (C Z=6, Cl Node 7 Z=17)",
        "• Node 9: C (Z=6, Chiral=0)  [Carbonyl C=O]",
        "• Node 10: O (Z=8, Chiral=0) [Carbonyl Oxygen]",
        "• Node 11: N (Z=7, Chiral=0) [Amide NH]",
        "• Nodes 12–15: Cyclopropyl Ring (C Z=6)"
    ]
    for i, line in enumerate(atom_lines):
        ax2.text(0.6, 8.65 - i*0.42, line, fontsize=9.0, color='#1e293b')

    ax2.text(0.5, 5.0, "Bond (Edge) Tokens in PyG ZINC: [Bond Type ID]",
             fontsize=10.5, fontweight='bold', color='#b45309')
    
    bond_lines = [
        "• PyG ZINC active bond types: 1 = Single, 2 = Double, 3 = Triple",
        "• GIN Encoder uses Embedding(5, 64) (Index 0 null, Index 4 reserved)",
        "• Molecule 2 contains 17 bonds (Single bonds: type 1, C=O: type 2)"
    ]
    for i, line in enumerate(bond_lines):
        ax2.text(0.6, 4.45 - i*0.42, line, fontsize=9.0, color='#1e293b')

    scaffold_box = FancyBboxPatch((0.4, 0.35), 9.2, 2.6, boxstyle="round,pad=0.06,rounding_size=0.10",
                                  facecolor='#fef3c7', edgecolor='#f59e0b', linewidth=1.5)
    ax2.add_patch(scaffold_box)
    ax2.text(5.0, 2.65, "Stored Canonical Bemis-Murcko Scaffold Key", ha='center', va='center',
            fontsize=10.5, fontweight='bold', color='#92400e')
    ax2.text(5.0, 2.15, "Scaffold SMILES: O=C(NCC1CC1)NCc1ccc(Cl)cc1", ha='center', va='center',
            fontsize=9.8, fontweight='bold', color='#78350f')
    ax2.text(5.0, 1.65, "• Preserves: Chlorophenyl ring, cyclopropyl ring, and linking amide", ha='center', va='center',
            fontsize=8.8, color='#78350f')
    ax2.text(5.0, 1.15, "• Stripped: Terminal methyl group on amine nitrogen (per Murcko rules)", ha='center', va='center',
            fontsize=8.8, fontstyle='italic', color='#78350f')
    ax2.text(5.0, 0.68, "• Retrieval Relevance: Matches corpus items sharing this identical non-empty key", ha='center', va='center',
            fontsize=8.8, fontweight='bold', color='#92400e')

    plt.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.05, wspace=0.10)
    out_path = os.path.join(FIGURES_DIR, "fig3_molecular_graph.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated {out_path}")

def generate_fig4_performance_comparison():
    """Generates Figure 6: Dual/grouped bar chart comparing MAP@10 and Recall@10 across all 7 systems (Non-overlapping x-ticks)."""
    systems = [
        "ECFP4 Tanimoto",
        "ECFP4 Jaccard",
        "GIN CE",
        "GIN SupCon-Aug",
        "GIN PA Neg-Rep",
        "GIN PA Excluded",
        "GIN SupCon"
    ]

    cov_map = [0.2294, 0.2304, 0.2423, 0.2928, 0.5418, 0.5593, 0.5996]
    cov_rec = [0.3116, 0.3109, 0.1911, 0.3663, 0.5127, 0.5244, 0.5939]
    all_map = [0.0631, 0.0634, 0.0666, 0.0805, 0.1490, 0.1538, 0.1649]
    all_rec = [0.0857, 0.0855, 0.0526, 0.1007, 0.1410, 0.1442, 0.1633]

    x = np.arange(len(systems))
    width = 0.38

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 5.2), dpi=300)

    bars1 = ax1.bar(x - width/2, cov_map, width, label='MAP@10', color='#1d4ed8', edgecolor='#1e3a8a', linewidth=1)
    bars2 = ax1.bar(x + width/2, cov_rec, width, label='Recall@10', color='#0d9488', edgecolor='#115e59', linewidth=1)
    ax1.set_title(r"(a) Covered Test Queries ($N = 275$ with $\geq 1$ Match)",
                  fontsize=10.5, fontweight='bold', color='#0b1d3a', pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(systems, fontsize=8.0, rotation=35, ha='right')
    ax1.set_ylabel("Retrieval Score", fontsize=9.5, fontweight='bold', color='#1e293b')
    ax1.set_ylim(0, 0.72)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    ax1.legend(loc='upper left', frameon=True, fontsize=8.5)

    best_idx = 6
    bars1[best_idx].set_edgecolor('#b91c1c')
    bars1[best_idx].set_linewidth(2)
    bars2[best_idx].set_edgecolor('#b91c1c')
    bars2[best_idx].set_linewidth(2)

    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.012, f"{yval:.4f}", ha='center', va='bottom', fontsize=6.8, rotation=90)
    for bar in bars2:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.012, f"{yval:.4f}", ha='center', va='bottom', fontsize=6.8, rotation=90)

    bars3 = ax2.bar(x - width/2, all_map, width, label='MAP@10', color='#6366f1', edgecolor='#4338ca', linewidth=1)
    bars4 = ax2.bar(x + width/2, all_rec, width, label='Recall@10', color='#14b8a6', edgecolor='#0f766e', linewidth=1)
    ax2.set_title("(b) Full Test Set ($N = 1,000$, 725 Uncovered Zeroed)",
                  fontsize=10.5, fontweight='bold', color='#0b1d3a', pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(systems, fontsize=8.0, rotation=35, ha='right')
    ax2.set_ylabel("Retrieval Score", fontsize=9.5, fontweight='bold', color='#1e293b')
    ax2.set_ylim(0, 0.22)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    ax2.legend(loc='upper left', frameon=True, fontsize=8.5)

    for bar in bars3:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.004, f"{yval:.4f}", ha='center', va='bottom', fontsize=6.8, rotation=90)
    for bar in bars4:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.004, f"{yval:.4f}", ha='center', va='bottom', fontsize=6.8, rotation=90)

    fig.suptitle("Comparative Evaluation Across Classical and Learned Retrieval Architectures",
                 fontsize=12.0, fontweight='bold', color='#0b1d3a', y=0.98)
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    out_path = os.path.join(FIGURES_DIR, "fig4_performance_comparison.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated {out_path}")

def generate_fig5_evaluation_coverage():
    """Generates Figure 3: Donut/pie chart illustrating 27.5% covered vs 72.5% uncovered queries (Fixed card boundaries, perfectly centered)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 5.2), dpi=300, gridspec_kw={'width_ratios': [1, 1.55]})

    labels = ['Covered Queries\n(>= 1 Corpus Match)\nN = 275 (27.5%)',
              'Uncovered Queries\n(0 Corpus Matches)\nN = 725 (72.5%)']
    sizes = [275, 725]
    colors = ['#0284c7', '#cbd5e1']
    explode = (0.06, 0)

    wedges, texts, autotexts = ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
                                      autopct='%1.1f%%', pctdistance=0.75, startangle=140,
                                      wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2),
                                      textprops={'fontsize': 9.0, 'color': '#0f172a', 'fontweight': 'bold'})
    for autotext in autotexts:
        autotext.set_color('#ffffff')
        autotext.set_fontsize(10.0)
        autotext.set_weight('bold')

    ax1.set_title("Test Query Exact Scaffold Coverage\n(1,000 Test Queries vs 11,000 Corpus)",
                  fontsize=11.5, fontweight='bold', color='#0b1d3a', pad=14)

    center_circle = plt.Circle((0,0), 0.52, fc='white')
    ax1.add_patch(center_circle)
    ax1.text(0, 0.08, "Total Queries", ha='center', va='center', fontsize=9.2, color='#64748b')
    ax1.text(0, -0.12, "1,000", ha='center', va='center', fontsize=15.5, fontweight='bold', color='#0f172a')

    # ax2 coordinate system: x in [0, 10], y in [0, 10]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')

    # Card box: x from 0.2 to 9.8 (width 9.6), y from 1.30 to 9.65 (height 8.35)
    card = FancyBboxPatch((0.2, 1.30), 9.6, 8.35, boxstyle="round,pad=0.08,rounding_size=0.14",
                          facecolor='#f8fafc', edgecolor='#64748b', linewidth=1.6)
    ax2.add_patch(card)

    # Card Title with ample padding inside
    ax2.text(5.0, 9.18, "Methodological Significance of Query Coverage", ha='center', va='center',
             fontsize=11.5, fontweight='bold', color='#0b1d3a')

    card_sections = [
        ("The 'All-Queries' Upper Bound Ceiling:", True, '#1e3a8a', 9.8, 0.52),
        ("Because 725 test molecules possess Murcko scaffolds that appear zero times in", False, '#334155', 8.5, 0.48),
        ("the 11,000-molecule corpus, no retrieval system can retrieve a true positive.", False, '#334155', 8.5, 0.48),
        ("Consequently, the mathematical upper bound on the full test set is bounded by:", False, '#334155', 8.5, 0.48),
        ("Max Possible MAP = 0.2750  |  Max Possible Recall = 0.2750", True, '#0284c7', 9.5, 0.72),
        ("Why Evaluating Covered Queries (N = 275) Is Essential:", True, '#0f766e', 9.8, 0.52),
        ("1. Isolates Model Quality: Measures precision and ranking strictly where ground", False, '#334155', 8.5, 0.48),
        ("   truth targets actually exist in the searchable candidate pool.", False, '#334155', 8.5, 0.50),
        ("2. Meaningful Separation: GIN-SupCon achieves MAP@10 of 0.5996 on covered", False, '#334155', 8.5, 0.48),
        ("   queries vs. 0.2294 for ECFP4 (+161% gain). On the full set, dilution dampens", False, '#334155', 8.5, 0.48),
        ("   these figures to 0.1649 vs 0.0631 due to the zero-target queries.", False, '#334155', 8.5, 0.50),
        ("3. Eliminates Zero Inflation: Prevents penalizing neural embedding methods for", False, '#334155', 8.5, 0.48),
        ("   molecules that have no corresponding scaffold in the library.", False, '#334155', 8.5, 0.48)
    ]

    y_pos = 8.42
    for line, is_bold, col, fs, step in card_sections:
        weight = 'bold' if is_bold else 'normal'
        ax2.text(0.60, y_pos, line, fontsize=fs, fontweight=weight, color=col, va='center')
        y_pos -= step

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig5_evaluation_coverage.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated {out_path}")

def main():
    print("Generating Figure 1: Retrieval Tradeoff...")
    generate_fig0_tradeoff()
    print("Generating Figure 4: System Workflow (Clean 3-Stage Pipeline)...")
    generate_fig1_system_workflow()
    print("Generating Figure 5: GIN Architecture (Enlarged Equations & Flow)...")
    generate_fig2_gin_architecture()
    print("Generating Figure 2: Molecular Graph (Balanced Text & Image)...")
    generate_fig3_molecular_graph()
    print("Generating Figure 6: Performance Comparison...")
    generate_fig4_performance_comparison()
    print("Generating Figure 3: Evaluation Coverage...")
    generate_fig5_evaluation_coverage()
    print("All figures generated successfully in figures/")

if __name__ == "__main__":
    main()
