"""
set_hashing.py - MinHash and LSH Banding Implementation for Molecular Shingles
CS328: Introduction to Data Science
Author: Laxmidhar Panda (24110185)
"""

from typing import List, Set, Dict, Tuple, Optional
from collections import defaultdict
import numpy as np


class MinHasher:
    """
    Min-wise Independent Permutations (MinHash) implementation using 2-universal hashing.
    Computes k-dimensional signatures for discrete sets to estimate Jaccard similarity.
    
    Collision identity: Pr[h_min(A) == h_min(B)] = J(A, B)
    """

    def __init__(self, num_hashes: int = 128, prime: int = 4294967311, seed: int = 42):
        """
        Initialize the universal hash functions: h_i(x) = (a_i * x + b_i) mod prime
        
        Args:
            num_hashes (k): Number of independent permutations/hashes.
            prime: Prime number strictly greater than 2^32 - 1.
            seed: Random seed for deterministic reproducibility.
        """
        self.num_hashes = num_hashes
        self.prime = prime
        self.rng = np.random.default_rng(seed)

        # Generate k random pairs (a_i, b_i)
        # a_i in [1, prime - 1], b_i in [0, prime - 1]
        self.a = self.rng.integers(1, self.prime, size=(1, self.num_hashes), dtype=np.int64)
        self.b = self.rng.integers(0, self.prime, size=(1, self.num_hashes), dtype=np.int64)

    def compute_signature(self, shingle_set: Set[int]) -> np.ndarray:
        """
        Compute the k-dimensional MinHash signature for a single molecular shingle set.
        
        Args:
            shingle_set: Set of integer shingle/substructure IDs.
            
        Returns:
            np.ndarray: 1D array of length k containing min-hash values (dtype=int64).
        """
        if not shingle_set:
            # Handle empty shingle set safely
            return np.zeros(self.num_hashes, dtype=np.int64)

        # Vectorized calculation:
        # shingles: (m, 1), a: (1, k), b: (1, k) -> hashes: (m, k)
        shingles = np.array(list(shingle_set), dtype=np.int64).reshape(-1, 1)
        # Ensure non-negative modulo
        shingles = np.bitwise_and(shingles, 0xFFFFFFFF)  # mask to uint32 space
        hashes = (shingles * self.a + self.b) % self.prime
        # Take column-wise minimum
        return np.min(hashes, axis=0)

    def compute_signature_matrix(self, shingle_sets: List[Set[int]]) -> np.ndarray:
        """
        Compute MinHash signature matrix for a list of molecules.
        
        Args:
            shingle_sets: List of shingle sets.
            
        Returns:
            np.ndarray: 2D array of shape (k, N) where each column is a molecule's signature.
        """
        n_items = len(shingle_sets)
        sig_matrix = np.empty((self.num_hashes, n_items), dtype=np.int64)
        for i, s_set in enumerate(shingle_sets):
            sig_matrix[:, i] = self.compute_signature(s_set)
        return sig_matrix

    @staticmethod
    def estimate_jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
        """
        Unbiased estimate of Jaccard similarity: fraction of matching hash slots.
        """
        return float(np.mean(sig_a == sig_b))


class LSHBandingIndex:
    """
    Locality-Sensitive Hashing (LSH) Banding Index.
    Partitions k signature rows into b bands of r rows (k = b * r).
    Surfaces candidate near-neighbors with probability P(s) = 1 - (1 - s^r)^b.
    """

    def __init__(self, b: int = 32, r: int = 4):
        """
        Initialize banding parameters.
        
        Args:
            b: Number of bands.
            r: Number of rows per band (k = b * r).
        """
        self.b = b
        self.r = r
        self.k = b * r
        # Phase transition threshold where P(t) = 0.5
        self.threshold = (1.0 / self.b) ** (1.0 / self.r)

        # Inverted index buckets per band: band_idx -> {hash_tuple: [doc_ids]}
        self.tables: List[Dict[Tuple[int, ...], List[int]]] = [defaultdict(list) for _ in range(self.b)]
        self.corpus_signatures: Optional[np.ndarray] = None
        self.num_indexed: int = 0

    def index(self, signature_matrix: np.ndarray):
        """
        Build the inverted hash bucket index from the signature matrix.
        
        Args:
            signature_matrix: 2D array of shape (k, N).
        """
        assert signature_matrix.shape[0] == self.k, (
            f"Signature matrix rows ({signature_matrix.shape[0]}) must equal b * r ({self.k})"
        )
        self.corpus_signatures = signature_matrix
        self.num_indexed = signature_matrix.shape[1]
        self.tables = [defaultdict(list) for _ in range(self.b)]

        for band_idx in range(self.b):
            start_row = band_idx * self.r
            end_row = start_row + self.r
            band_data = signature_matrix[start_row:end_row, :]  # shape: (r, N)
            table = self.tables[band_idx]

            for doc_id in range(self.num_indexed):
                bucket_key = tuple(band_data[:, doc_id].tolist())
                table[bucket_key].append(doc_id)

    def get_candidates(self, query_sig: np.ndarray) -> Set[int]:
        """
        Return the set of candidate document IDs colliding with query in at least one band.
        
        Args:
            query_sig: 1D array of length k.
            
        Returns:
            Set[int]: Union of all colliding candidate IDs.
        """
        candidates: Set[int] = set()
        for band_idx in range(self.b):
            start_row = band_idx * self.r
            end_row = start_row + self.r
            bucket_key = tuple(query_sig[start_row:end_row].tolist())
            matches = self.tables[band_idx].get(bucket_key)
            if matches:
                candidates.update(matches)
        return candidates

    def query_top_k(
        self,
        query_sig: np.ndarray,
        k: int = 10,
        corpus_sigs: Optional[np.ndarray] = None
    ) -> List[Tuple[int, float]]:
        """
        Sublinear retrieval: find candidates via banding buckets, then re-rank
        candidates by estimated Jaccard similarity.
        
        Returns:
            List of (doc_id, estimated_jaccard) sorted descending by similarity.
        """
        if corpus_sigs is None:
            corpus_sigs = self.corpus_signatures
        assert corpus_sigs is not None, "Corpus signatures must be provided or indexed."

        candidates = self.get_candidates(query_sig)
        if not candidates:
            # Fallback if no bucket collision: sample small random or empty
            return []

        cand_list = list(candidates)
        cand_sigs = corpus_sigs[:, cand_list]  # (k, len(cand_list))
        # Vectorized similarity comparison:
        matches = (cand_sigs == query_sig[:, np.newaxis])  # (k, num_cands)
        sims = np.mean(matches, axis=0)  # (num_cands,)

        if len(cand_list) <= k:
            top_indices = np.argsort(-sims)
        else:
            # Partial sort for efficiency
            top_k_partition = np.argpartition(-sims, k)[:k]
            top_indices = top_k_partition[np.argsort(-sims[top_k_partition])]

        return [(cand_list[idx], float(sims[idx])) for idx in top_indices]

    @staticmethod
    def theoretical_collision_prob(s: np.ndarray, b: int = 32, r: int = 4) -> np.ndarray:
        """
        Theoretical S-curve formula: P(s) = 1 - (1 - s^r)^b
        """
        return 1.0 - (1.0 - np.power(s, r)) ** b
