"""
vector_search.py - Sublinear Vector Search Serving (SimHash LSH and FAISS HNSW)
CS328: Introduction to Data Science
Author: Laxmidhar Panda (24110185)
"""

from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class SimHashIndex:
    """
    Hyperplane Locality-Sensitive Hashing (SimHash / Random Projection LSH).
    Maps continuous d-dimensional vectors to l-bit signatures using random Gaussian hyperplanes.
    Collision probability between x and y equals: 1 - theta(x, y) / pi.
    Applies banding (b bands, r rows) to retrieve nearest neighbors sublinearly.
    """

    def __init__(self, d: int = 256, l: int = 128, b: int = 32, r: int = 4, seed: int = 42):
        assert l == b * r, f"l ({l}) must equal b * r ({b * r})"
        self.d = d
        self.l = l
        self.b = b
        self.r = r
        self.rng = np.random.default_rng(seed)

        # Draw random hyperplanes G ~ N(0, I) of shape (l, d)
        self.hyperplanes = self.rng.standard_normal(size=(self.l, self.d)).astype(np.float32)
        # Normalize hyperplanes
        norms = np.linalg.norm(self.hyperplanes, axis=1, keepdims=True)
        self.hyperplanes = self.hyperplanes / np.maximum(norms, 1e-12)

        self.tables: List[Dict[Tuple[int, ...], List[int]]] = [defaultdict(list) for _ in range(self.b)]
        self.corpus_embeddings: Optional[np.ndarray] = None
        self.corpus_bits: Optional[np.ndarray] = None
        self.num_indexed: int = 0

    def compute_bit_signatures(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Project embeddings onto random hyperplanes and take signum:
        h(x) = (x @ G.T >= 0) -> shape (N, l) as uint8
        """
        projections = np.dot(embeddings, self.hyperplanes.T)  # (N, l)
        return (projections >= 0).astype(np.uint8)

    def index(self, embeddings: np.ndarray):
        """
        Index continuous embeddings into SimHash banding tables.
        embeddings: (N, d) float32 (assumed L2-normalized)
        """
        self.corpus_embeddings = embeddings.astype(np.float32)
        self.num_indexed = len(embeddings)
        self.corpus_bits = self.compute_bit_signatures(self.corpus_embeddings)
        self.tables = [defaultdict(list) for _ in range(self.b)]

        for band_idx in range(self.b):
            start = band_idx * self.r
            end = start + self.r
            band_data = self.corpus_bits[:, start:end]
            table = self.tables[band_idx]

            for doc_id in range(self.num_indexed):
                bucket_key = tuple(band_data[doc_id].tolist())
                table[bucket_key].append(doc_id)

    def get_candidates(self, query_bits: np.ndarray) -> Set[int]:
        candidates: Set[int] = set()
        for band_idx in range(self.b):
            start = band_idx * self.r
            end = start + self.r
            bucket_key = tuple(query_bits[start:end].tolist())
            matches = self.tables[band_idx].get(bucket_key)
            if matches:
                candidates.update(matches)
        return candidates

    def search(self, query_vector: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sublinear search: find candidate bucket collisions, then re-rank via exact inner product.
        Returns:
            (top_scores, top_indices)
        """
        q_vec = query_vector.reshape(1, -1).astype(np.float32)
        q_bits = self.compute_bit_signatures(q_vec)[0]
        candidates = self.get_candidates(q_bits)

        if not candidates:
            # Fallback: if no bucket collision, search a random sample of 50 or return empty
            return np.array([]), np.array([])

        cand_list = list(candidates)
        cand_vecs = self.corpus_embeddings[cand_list]  # (C, d)
        scores = np.dot(cand_vecs, q_vec.T).squeeze(-1)  # cosine similarity

        if len(cand_list) <= k:
            top_k_idx = np.argsort(-scores)
        else:
            partition = np.argpartition(-scores, k)[:k]
            top_k_idx = partition[np.argsort(-scores[partition])]

        return scores[top_k_idx], np.array(cand_list)[top_k_idx]


class ContinuousHNSWIndex:
    """
    FAISS Hierarchical Navigable Small World (HNSW) index for continuous embeddings.
    Uses Inner Product metric on L2-normalized embeddings, guaranteeing cosine equivalence.
    """

    def __init__(self, d: int = 256, M: int = 32, efConstruction: int = 200, efSearch: int = 64):
        self.d = d
        self.M = M
        self.efConstruction = efConstruction
        self.efSearch = efSearch
        self.faiss_index = None
        self.corpus_embeddings = None

    def index(self, embeddings: np.ndarray):
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.corpus_embeddings = embeddings
        if FAISS_AVAILABLE:
            self.faiss_index = faiss.IndexHNSWFlat(self.d, self.M, faiss.METRIC_INNER_PRODUCT)
            self.faiss_index.hnsw.efConstruction = self.efConstruction
            self.faiss_index.hnsw.efSearch = self.efSearch
            self.faiss_index.add(embeddings)
        else:
            print("[Warning] FAISS not available; falling back to exact scan.")

    def set_ef_search(self, ef_search: int):
        self.efSearch = ef_search
        if FAISS_AVAILABLE and self.faiss_index is not None:
            self.faiss_index.hnsw.efSearch = ef_search

    def search(self, query_vector: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        q_vec = np.ascontiguousarray(query_vector.reshape(1, -1), dtype=np.float32)
        if FAISS_AVAILABLE and self.faiss_index is not None:
            distances, indices = self.faiss_index.search(q_vec, k)
            return distances[0], indices[0]
        else:
            # Exact inner product fallback
            scores = np.dot(self.corpus_embeddings, q_vec.T).squeeze(-1)
            top_idx = np.argsort(-scores)[:k]
            return scores[top_idx], top_idx


class TwoStageRetriever:
    """
    Two-Stage Retrieval Pipeline:
    - Stage 1: Fast ANN candidate generation using HNSW (shortlist of size M, e.g. M=50).
    - Stage 2: Fine-grained exact metric reranking over the shortlist to produce final top-K.
    """

    def __init__(self, first_stage_index: ContinuousHNSWIndex, shortlist_m: int = 50):
        self.first_stage = first_stage_index
        self.shortlist_m = shortlist_m

    def search(self, query_vector: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        # Stage 1: retrieve shortlist
        scores_m, indices_m = self.first_stage.search(query_vector, k=self.shortlist_m)
        if len(indices_m) == 0:
            return np.array([]), np.array([])

        # Stage 2: exact re-scoring on the shortlist
        q_vec = query_vector.reshape(1, -1).astype(np.float32)
        cand_vecs = self.first_stage.corpus_embeddings[indices_m]
        exact_scores = np.dot(cand_vecs, q_vec.T).squeeze(-1)

        # Sort descending
        top_k_sub = np.argsort(-exact_scores)[:k]
        return exact_scores[top_k_sub], indices_m[top_k_sub]



class BinaryHNSWIndex:
    """
    FAISS Binary HNSW Index for discrete folded 1024-bit ECFP fingerprints.
    Uses Hamming distance to navigate proximity graph over discrete representations.
    """

    def __init__(self, n_bits: int = 1024, M: int = 32):
        self.n_bits = n_bits
        self.n_bytes = n_bits // 8
        self.M = M
        self.faiss_index = None
        self.corpus_packed = None

    @staticmethod
    def pack_bits(bit_matrix: np.ndarray) -> np.ndarray:
        """
        Convert (N, 1024) boolean/uint8 bit matrix into (N, 128) uint8 byte array.
        """
        return np.packbits(bit_matrix, axis=1)

    def index(self, bit_matrix: np.ndarray):
        packed = self.pack_bits(bit_matrix)
        self.corpus_packed = packed
        if FAISS_AVAILABLE:
            try:
                self.faiss_index = faiss.IndexBinaryHNSW(self.n_bits, self.M)
                self.faiss_index.add(packed)
            except Exception:
                # If binary HNSW is unavailable in cpu build, use IndexBinaryFlat
                self.faiss_index = faiss.IndexBinaryFlat(self.n_bits)
                self.faiss_index.add(packed)
        else:
            print("[Warning] FAISS not available; using fast popcount Hamming fallback.")

    def search(self, query_bits: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        q_packed = self.pack_bits(query_bits.reshape(1, -1))
        if FAISS_AVAILABLE and self.faiss_index is not None:
            distances, indices = self.faiss_index.search(q_packed, k)
            # FAISS returns Hamming distance (lower is closer).
            # Convert to similarity: 1.0 - (dist / n_bits)
            sims = 1.0 - (distances[0].astype(np.float32) / self.n_bits)
            return sims, indices[0]
        else:
            # Vectorized XOR popcount
            xor_result = np.bitwise_xor(self.corpus_packed, q_packed)
            # Count bits set
            hamming_dists = np.sum(np.unpackbits(xor_result, axis=1), axis=1)
            top_idx = np.argsort(hamming_dists)[:k]
            sims = 1.0 - (hamming_dists[top_idx].astype(np.float32) / self.n_bits)
            return sims, top_idx
