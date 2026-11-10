"""
train_gin.py - Train 4-Layer GIN with SupCon Loss and Extract Hyperspherical Embeddings
CS328: Introduction to Data Science
Author: Laxmidhar Panda (24110185)
"""

import os
import sys
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.gin_encoder import GINEncoder, SupConLoss


class LabeledGraphDataset(torch.utils.data.Dataset):
    """Wraps PyG Dataset to inject precomputed structural class labels into mini-batches."""
    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.labels = labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        data.label = torch.tensor(self.labels[idx], dtype=torch.long)
        return data


def train_gin(
    data_root: str = "data/ZINC",
    cache_path: str = "data/processed_cache.pkl",
    artifacts_dir: str = "artifacts",
    batch_size: int = 128,
    epochs: int = 25,
    lr: float = 1e-3,
    temperature: float = 0.07,
    seed: int = 42
):
    print("=== Phase 2: Neural Representation Learning (GIN + SupCon) ===")
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    os.makedirs(artifacts_dir, exist_ok=True)

    # 1. Load structural labels from cache
    print("Loading preprocessed labels...")
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)

    train_labels = cache["train_labels"]
    val_labels = cache["val_labels"]
    query_labels = cache["query_labels"]

    # 2. Load PyG ZINC datasets and wrap with labels
    print("Loading PyG ZINC datasets...")
    train_ds = ZINC(root=data_root, subset=True, split="train")
    val_ds = ZINC(root=data_root, subset=True, split="val")
    test_ds = ZINC(root=data_root, subset=True, split="test")

    train_wrapped = LabeledGraphDataset(train_ds, train_labels)
    val_wrapped = LabeledGraphDataset(val_ds, val_labels)
    test_wrapped = LabeledGraphDataset(test_ds, query_labels)

    train_loader = DataLoader(train_wrapped, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_wrapped, batch_size=batch_size, shuffle=False)

    # 3. Model, Loss, Optimizer
    model = GINEncoder(
        num_atom_types=28,
        num_bond_types=5,
        hidden_dim=64,
        proj_dim=256,
        num_layers=4
    ).to(device)

    criterion = SupConLoss(temperature=temperature).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_model_path = os.path.join(artifacts_dir, "gin_best_model.pt")

    # 4. Training Loop
    print(f"\nStarting SupCon Training for {epochs} epochs (Batch Size={batch_size}, tau={temperature})...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        num_train_batches = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Forward pass: outputs strictly L2-normalized 256-D vectors
            z = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(z, batch.label)

            if not torch.isnan(loss) and loss.item() > 0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_train_loss += loss.item()
                num_train_batches += 1

        scheduler.step()
        avg_train_loss = total_train_loss / max(num_train_batches, 1)

        # Validation
        model.eval()
        total_val_loss = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                z = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                val_loss = criterion(z, batch.label)
                if not torch.isnan(val_loss):
                    total_val_loss += val_loss.item()
                    num_val_batches += 1

        avg_val_loss = total_val_loss / max(num_val_batches, 1)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train SupCon Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)

    print(f"\nTraining Complete. Best model saved to: {best_model_path}")

    # 5. Extract embeddings for all 11,000 corpus graphs and 1,000 test queries
    print("\n--- Extracting Hyperspherical Continuous Embeddings ---")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    def extract_dataset_embeddings(dataset) -> np.ndarray:
        loader = DataLoader(dataset, batch_size=256, shuffle=False)
        embs = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                z = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                embs.append(z.cpu().numpy())
        return np.vstack(embs)

    print("Extracting train + val corpus embeddings (11,000 graphs)...")
    train_embs = extract_dataset_embeddings(train_ds)
    val_embs = extract_dataset_embeddings(val_ds)
    corpus_embeddings = np.vstack([train_embs, val_embs])

    print("Extracting test query embeddings (1,000 graphs)...")
    query_embeddings = extract_dataset_embeddings(test_ds)

    # 6. Integrity Check 2: Strictly L2-normalized on S^{255}
    print("\n=== Running Integrity Check 2: Hyperspherical Normalization ===")
    corpus_norms = np.linalg.norm(corpus_embeddings, axis=1)
    query_norms = np.linalg.norm(query_embeddings, axis=1)

    max_dev_corpus = float(np.max(np.abs(corpus_norms - 1.0)))
    max_dev_query = float(np.max(np.abs(query_norms - 1.0)))
    print(f"Corpus L2 Norm Max Deviation from 1.0: {max_dev_corpus:.2e}")
    print(f"Query  L2 Norm Max Deviation from 1.0: {max_dev_query:.2e}")

    assert max_dev_corpus < 1e-5, f"Integrity Check 2 Failed: Corpus vectors not normalized! (dev={max_dev_corpus})"
    assert max_dev_query < 1e-5, f"Integrity Check 2 Failed: Query vectors not normalized! (dev={max_dev_query})"
    print(">> INTEGRITY CHECK 2 PASSED: All GIN embeddings sit strictly on the unit hypersphere S^255.")

    # Save continuous embeddings
    corpus_emb_path = os.path.join(artifacts_dir, "corpus_embeddings.npy")
    query_emb_path = os.path.join(artifacts_dir, "query_embeddings.npy")
    np.save(corpus_emb_path, corpus_embeddings)
    np.save(query_emb_path, query_embeddings)
    print(f"Saved corpus embeddings ({corpus_embeddings.shape}) to {corpus_emb_path}")
    print(f"Saved query embeddings ({query_embeddings.shape}) to {query_emb_path}")


if __name__ == "__main__":
    train_gin()
