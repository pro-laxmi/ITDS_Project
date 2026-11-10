"""
data_ingestion.py - Ingest ZINC 12k Subset, RDKit Molecular Reconstruction, and ECFP4 Fingerprinting
CS328: Introduction to Data Science
Author: Laxmidhar Panda (24110185)
"""

import os
import pickle
from typing import List, Set, Tuple, Dict, Any, Optional
import numpy as np
import torch
from torch_geometric.datasets import ZINC
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

# Suppress RDKit verbose deprecation warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# ZINC atom tokens from raw atom_dict.pickle:
ZINC_ATOM_TOKENS = [
    'C', 'O', 'N', 'F', 'C H1', 'S', 'Cl', 'O -', 'N H1 +', 'Br',
    'N H3 +', 'N H2 +', 'N +', 'N -', 'S -', 'I', 'P', 'O H1 +',
    'N H1 -', 'O +', 'S +', 'P H1', 'P H2', 'C H2 -', 'P +',
    'S H1 +', 'C H1 -', 'P H1 +'
]

BOND_TYPES = {
    1: Chem.rdchem.BondType.SINGLE,
    2: Chem.rdchem.BondType.DOUBLE,
    3: Chem.rdchem.BondType.TRIPLE
}


def parse_token(t: str) -> Tuple[str, int]:
    """Parse symbol and formal charge from ZINC token string."""
    parts = t.split()
    sym = parts[0]
    charge = 0
    if '+' in t:
        charge = 1
    elif '-' in t:
        charge = -1
    return sym, charge


def pyg_to_rdkit_mol(data) -> Chem.Mol:
    """
    Reconstruct an RDKit Mol object from PyG graph data.
    """
    mol = Chem.RWMol()
    node_tokens = data.x.squeeze(-1).tolist()
    for tok_idx in node_tokens:
        tok_str = ZINC_ATOM_TOKENS[tok_idx] if tok_idx < len(ZINC_ATOM_TOKENS) else 'C'
        sym, charge = parse_token(tok_str)
        atom = Chem.Atom(sym)
        atom.SetFormalCharge(charge)
        mol.AddAtom(atom)

    edge_index = data.edge_index.tolist()
    edge_attr = data.edge_attr.tolist() if data.edge_attr is not None else [1] * len(edge_index[0])
    added = set()
    for u, v, b in zip(edge_index[0], edge_index[1], edge_attr):
        if u < v and (u, v) not in added:
            added.add((u, v))
            b_type = BOND_TYPES.get(b, Chem.rdchem.BondType.SINGLE)
            mol.AddBond(u, v, b_type)

    mol_obj = mol.GetMol()
    mol_obj.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol_obj)
    return mol_obj


def extract_ecfp4_for_mol(
    mol: Chem.Mol,
    radius: int = 2,
    n_bits: int = 1024
) -> Tuple[Set[int], np.ndarray]:
    """
    Extract sparse shingle set (unsigned 32-bit hashes) and 1024-bit folded binary vector.
    """
    info: Dict[int, Any] = {}
    fp_counts = AllChem.GetMorganFingerprint(mol, radius, bitInfo=info)
    shingles = set(int(k) for k in fp_counts.GetNonzeroElements().keys())

    bit_vect = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    bits = np.zeros(n_bits, dtype=np.uint8)
    for bit_idx in bit_vect.GetOnBits():
        bits[bit_idx] = 1

    return shingles, bits


def get_scaffold_class(mol: Chem.Mol, fallback_id: int) -> str:
    """Extract Bemis-Murcko scaffold or size-based cluster."""
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        if scaffold and len(scaffold) > 0:
            return scaffold
    except Exception:
        pass
    num_atoms = mol.GetNumAtoms() if mol is not None else 10
    return f"acyclic_bin_{num_atoms // 4}"


def assign_structural_classes(scaffold_keys: List[str], min_class_size: int = 3) -> np.ndarray:
    """
    Assign integer labels y in {0, ..., C-1}.
    """
    from collections import Counter
    counts = Counter(scaffold_keys)
    frequent = {k for k, v in counts.items() if v >= min_class_size}

    class_to_id = {k: i for i, k in enumerate(sorted(frequent))}
    overflow_offset = len(class_to_id)
    labels = np.empty(len(scaffold_keys), dtype=np.int64)

    for i, key in enumerate(scaffold_keys):
        if key in class_to_id:
            labels[i] = class_to_id[key]
        else:
            # Hash to pseudo-clusters of size ~30 to guarantee positive pairs
            labels[i] = overflow_offset + (hash(key) % 30)

    return labels


class ClassBalancedSampler(torch.utils.data.Sampler):
    """
    Class-balanced mini-batch sampler (matching FamilyBalancedSampler from reference paper).
    In each batch, samples P classes uniformly, and K instances per class.
    Batch size = P * K.
    Guarantees every anchor has exactly K - 1 in-batch positive instances.
    """

    def __init__(self, labels: np.ndarray, p_classes: int = 16, k_instances: int = 8, batches_per_epoch: int = 100):
        self.labels = np.array(labels)
        self.p_classes = p_classes
        self.k_instances = k_instances
        self.batch_size = p_classes * k_instances
        self.batches_per_epoch = batches_per_epoch

        # Index instances by class
        self.class_to_indices = {}
        for idx, lbl in enumerate(self.labels):
            if lbl not in self.class_to_indices:
                self.class_to_indices[lbl] = []
            self.class_to_indices[lbl].append(idx)

        # Classes with at least 2 instances can form positive pairs
        self.valid_classes = [c for c, idxs in self.class_to_indices.items() if len(idxs) >= 2]
        assert len(self.valid_classes) >= p_classes, f"Need at least {p_classes} classes, found {len(self.valid_classes)}"

    def __len__(self):
        return self.batches_per_epoch * self.batch_size

    def __iter__(self):
        for _ in range(self.batches_per_epoch):
            selected_classes = np.random.choice(self.valid_classes, size=self.p_classes, replace=False)
            batch = []
            for c in selected_classes:
                indices = self.class_to_indices[c]
                replace = len(indices) < self.k_instances
                sampled = np.random.choice(indices, size=self.k_instances, replace=replace)
                batch.extend(sampled.tolist())
            yield [int(x) for x in batch]


def prepare_and_cache_dataset(
    root: str = "data/ZINC",
    cache_path: str = "data/processed_cache.pkl"
) -> Dict[str, Any]:
    """
    Load PyG ZINC 12k dataset (train, val, test), extract RDKit features,
    assign structural classes, and cache the precomputed feature matrices.
    """
    if os.path.exists(cache_path):
        print(f"Loading cached dataset from {cache_path}...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print("Loading PyG ZINC subsets...")
    train_ds = ZINC(root=root, subset=True, split="train")
    val_ds = ZINC(root=root, subset=True, split="val")
    test_ds = ZINC(root=root, subset=True, split="test")

    print(f"Loaded: {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test graphs.")

    # 1. Corpus (train + val = 11,000 graphs)
    print("Converting corpus graphs to RDKit molecules and extracting ECFP4...")
    corpus_shingles: List[Set[int]] = []
    corpus_bits_list: List[np.ndarray] = []
    corpus_scaffolds: List[str] = []

    for i in tqdm(range(len(train_ds)), desc="Train"):
        mol = pyg_to_rdkit_mol(train_ds[i])
        shingles, bits = extract_ecfp4_for_mol(mol)
        scaffold = get_scaffold_class(mol, i)
        corpus_shingles.append(shingles)
        corpus_bits_list.append(bits)
        corpus_scaffolds.append(scaffold)

    for j in tqdm(range(len(val_ds)), desc="Val"):
        mol = pyg_to_rdkit_mol(val_ds[j])
        shingles, bits = extract_ecfp4_for_mol(mol)
        scaffold = get_scaffold_class(mol, len(train_ds) + j)
        corpus_shingles.append(shingles)
        corpus_bits_list.append(bits)
        corpus_scaffolds.append(scaffold)

    # 2. Queries (test = 1,000 graphs)
    print("Converting query graphs to RDKit molecules and extracting ECFP4...")
    query_shingles: List[Set[int]] = []
    query_bits_list: List[np.ndarray] = []
    query_scaffolds: List[str] = []

    for k in tqdm(range(len(test_ds)), desc="Test Queries"):
        mol = pyg_to_rdkit_mol(test_ds[k])
        shingles, bits = extract_ecfp4_for_mol(mol)
        scaffold = get_scaffold_class(mol, k)
        query_shingles.append(shingles)
        query_bits_list.append(bits)
        query_scaffolds.append(scaffold)

    # Assign structural classes jointly
    all_scaffolds = corpus_scaffolds + query_scaffolds
    all_labels = assign_structural_classes(all_scaffolds, min_class_size=3)

    corpus_labels = all_labels[:len(corpus_shingles)]
    query_labels = all_labels[len(corpus_shingles):]

    corpus_bits = np.vstack(corpus_bits_list)
    query_bits = np.vstack(query_bits_list)

    # Lightweight cache dictionary (under 15 MB)
    data_dict = {
        "corpus_shingles": corpus_shingles,
        "query_shingles": query_shingles,
        "corpus_bits": corpus_bits,
        "query_bits": query_bits,
        "corpus_labels": corpus_labels,
        "query_labels": query_labels,
        "train_labels": corpus_labels[:len(train_ds)],
        "val_labels": corpus_labels[len(train_ds):]
    }

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Data ingestion complete! Cache successfully written to {cache_path}")
    print(f"Unique classes: {len(np.unique(all_labels))}, Corpus bits shape: {corpus_bits.shape}, Query bits shape: {query_bits.shape}")
    return data_dict


if __name__ == "__main__":
    prepare_and_cache_dataset()
