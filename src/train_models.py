"""
train_models.py - Train GIN-CE, GIN-SupCon-Aug, and GIN-ProxyAnchor Models
CS328: Introduction to Data Science
Author: Laxmidhar Panda (24110185)
"""

import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.gin_encoder import (
    GINEncoder, GINClassifier, SupConLoss, ProxyAnchorLoss,
    drop_edges, mask_features
)
from src.data_ingestion import ClassBalancedSampler


class LabeledGraphDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.labels = labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        data.label = torch.tensor(self.labels[idx], dtype=torch.long)
        return data


def extract_embeddings(model: nn.Module, dataset, device: torch.device) -> np.ndarray:
    model.eval()
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    embs = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if hasattr(model, "encoder"):  # GINClassifier
                z = model.encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            else:
                z = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            embs.append(z.cpu().numpy())
    return np.vstack(embs)


def train_gin_ce(train_ds, val_ds, test_ds, num_classes: int, device: torch.device, epochs: int = 25):
    print("\n" + "="*60)
    print(" Training Model 1: GIN-CE (Cross-Entropy Baseline)")
    print("="*60)

    encoder = GINEncoder(num_atom_types=28, num_bond_types=5, hidden_dim=64, proj_dim=256, num_layers=4).to(device)
    model = GINClassifier(encoder, num_classes=num_classes, proj_dim=256).to(device)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_path = "artifacts/gin_ce_best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(logits, batch.label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss = total_loss / max(n_batches, 1)

        model.eval()
        v_loss = 0.0
        vn = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                v_loss += criterion(logits, batch.label).item()
                vn += 1
        val_loss = v_loss / max(vn, 1)

        if epoch % 5 == 0 or epoch == epochs:
            print(f"GIN-CE Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)

    print(f"GIN-CE Training Complete. Best model saved to: {best_path}")
    model.load_state_dict(torch.load(best_path, map_location=device))
    return model


def train_gin_supcon_aug(train_ds, val_ds, test_ds, train_labels, device: torch.device, epochs: int = 25):
    print("\n" + "="*60)
    print(" Training Model 2: GIN-SupCon-Aug (GraphCL Augmentations + Class-Balanced Sampling)")
    print("="*60)

    model = GINEncoder(num_atom_types=28, num_bond_types=5, hidden_dim=64, proj_dim=256, num_layers=4).to(device)
    criterion = SupConLoss(temperature=0.07).to(device)

    # Class-balanced mini-batch sampler: 16 classes x 8 instances = 128 batch size
    balanced_sampler = ClassBalancedSampler(train_labels, p_classes=16, k_instances=8, batches_per_epoch=75)
    train_loader = DataLoader(train_ds, batch_sampler=balanced_sampler)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_path = "artifacts/gin_supcon_aug_best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Apply GraphCL stochastic augmentations: edge dropout (p=0.1) and feature masking (p=0.1)
            aug_edge_index, aug_edge_attr = drop_edges(batch.edge_index, batch.edge_attr, p=0.1)
            aug_x = mask_features(batch.x, p=0.1)

            z = model(aug_x, aug_edge_index, aug_edge_attr, batch.batch)
            loss = criterion(z, batch.label)

            if not torch.isnan(loss) and loss.item() > 0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

        scheduler.step()
        train_loss = total_loss / max(n_batches, 1)

        model.eval()
        v_loss = 0.0
        vn = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                z = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                l = criterion(z, batch.label)
                if not torch.isnan(l):
                    v_loss += l.item()
                    vn += 1
        val_loss = v_loss / max(vn, 1)

        if epoch % 5 == 0 or epoch == epochs:
            print(f"GIN-SupCon-Aug Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)

    print(f"GIN-SupCon-Aug Training Complete. Best model saved to: {best_path}")
    model.load_state_dict(torch.load(best_path, map_location=device))
    return model


def train_gin_proxyanchor(train_ds, val_ds, test_ds, num_classes: int, device: torch.device, epochs: int = 25):
    print("\n" + "="*60)
    print(" Training Model 3: GIN-ProxyAnchor (Metric Learning via Class Proxies)")
    print("="*60)

    model = GINEncoder(num_atom_types=28, num_bond_types=5, hidden_dim=64, proj_dim=256, num_layers=4).to(device)
    criterion = ProxyAnchorLoss(num_classes=num_classes, embedding_dim=256, margin=0.1, alpha=32.0).to(device)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=1e-3,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_path = "artifacts/gin_proxyanchor_best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            z = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(z, batch.label)

            if not torch.isnan(loss) and loss.item() > 0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

        scheduler.step()
        train_loss = total_loss / max(n_batches, 1)

        model.eval()
        v_loss = 0.0
        vn = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                z = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                l = criterion(z, batch.label)
                if not torch.isnan(l):
                    v_loss += l.item()
                    vn += 1
        val_loss = v_loss / max(vn, 1)

        if epoch % 5 == 0 or epoch == epochs:
            print(f"GIN-ProxyAnchor Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)

    print(f"GIN-ProxyAnchor Training Complete. Best model saved to: {best_path}")
    model.load_state_dict(torch.load(best_path, map_location=device))
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Multi-Model Training Suite on device: {device}")

    with open("data/processed_cache.pkl", "rb") as f:
        cache = pickle.load(f)

    train_labels = cache["train_labels"]
    val_labels = cache["val_labels"]
    query_labels = cache["query_labels"]
    num_classes = len(np.unique(cache["corpus_labels"]))

    print(f"Loaded dataset: 10,000 train, 1,000 val, 1,000 test across {num_classes} classes.")

    train_raw = ZINC(root="data/ZINC", subset=True, split="train")
    val_raw = ZINC(root="data/ZINC", subset=True, split="val")
    test_raw = ZINC(root="data/ZINC", subset=True, split="test")

    train_ds = LabeledGraphDataset(train_raw, train_labels)
    val_ds = LabeledGraphDataset(val_raw, val_labels)
    test_ds = LabeledGraphDataset(test_raw, query_labels)

    # 1. Train GIN-CE (skip if already trained and saved)
    if not (os.path.exists("artifacts/embs_ce_corpus.npy") and os.path.exists("artifacts/embs_ce_query.npy")):
        model_ce = train_gin_ce(train_ds, val_ds, test_ds, num_classes, device, epochs=25)
        embs_ce_train = extract_embeddings(model_ce, train_raw, device)
        embs_ce_val = extract_embeddings(model_ce, val_raw, device)
        embs_ce_corpus = np.vstack([embs_ce_train, embs_ce_val])
        embs_ce_query = extract_embeddings(model_ce, test_raw, device)
        np.save("artifacts/embs_ce_corpus.npy", embs_ce_corpus)
        np.save("artifacts/embs_ce_query.npy", embs_ce_query)
        print(f"Saved GIN-CE embeddings: Corpus {embs_ce_corpus.shape}, Query {embs_ce_query.shape}")
    else:
        print("Found existing GIN-CE embeddings in artifacts/. Skipping re-training.")

    # 2. Train GIN-SupCon-Aug
    model_aug = train_gin_supcon_aug(train_ds, val_ds, test_ds, train_labels, device, epochs=25)
    embs_aug_train = extract_embeddings(model_aug, train_raw, device)
    embs_aug_val = extract_embeddings(model_aug, val_raw, device)
    embs_aug_corpus = np.vstack([embs_aug_train, embs_aug_val])
    embs_aug_query = extract_embeddings(model_aug, test_raw, device)
    np.save("artifacts/embs_supcon_aug_corpus.npy", embs_aug_corpus)
    np.save("artifacts/embs_supcon_aug_query.npy", embs_aug_query)
    print(f"Saved GIN-SupCon-Aug embeddings: Corpus {embs_aug_corpus.shape}, Query {embs_aug_query.shape}")

    # 3. Train GIN-ProxyAnchor
    model_pa = train_gin_proxyanchor(train_ds, val_ds, test_ds, num_classes, device, epochs=25)
    embs_pa_train = extract_embeddings(model_pa, train_raw, device)
    embs_pa_val = extract_embeddings(model_pa, val_raw, device)
    embs_pa_corpus = np.vstack([embs_pa_train, embs_pa_val])
    embs_pa_query = extract_embeddings(model_pa, test_raw, device)
    np.save("artifacts/embs_proxyanchor_corpus.npy", embs_pa_corpus)
    np.save("artifacts/embs_proxyanchor_query.npy", embs_pa_query)
    print(f"Saved GIN-ProxyAnchor embeddings: Corpus {embs_pa_corpus.shape}, Query {embs_pa_query.shape}")

    print("\nAll 3 models trained and embeddings extracted successfully!")


if __name__ == "__main__":
    main()
