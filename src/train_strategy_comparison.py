"""
train_strategy_comparison.py - Empirically compare Training Label Strategies:
  Strategy A: Legacy 336 classes with 30 overflow pseudo-clusters (derived from all 12k).
  Strategy B: Train-only exact scaffolds (244 frequent classes >= 3 in train, rare molecules unclustered).
  Strategy C: Train-only rare-class strategy (610 classes with >= 2 instances in train, rare singletons unclustered).

Evaluates exact canonical Bemis-Murcko scaffold retrieval on N=275 covered queries and all N=1000 queries.
"""

import os
import sys
import time
import pickle
import json
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.gin_encoder import GINEncoder, ProxyAnchorLoss
from src.evaluate_exact_scaffolds import compute_metrics_for_gt

class ProxyAnchorWithUnlabeled(nn.Module):
    """
    Proxy-Anchor loss that gracefully handles unlabeled/rare samples (label == -1).
    Samples with label == -1 do not pull any positive proxy, but act as negative
    repellers for all class proxies.
    """
    def __init__(self, num_classes: int, embedding_dim: int = 256, margin: float = 0.1, alpha: float = 32.0):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.alpha = alpha
        self.proxies = nn.Parameter(torch.randn(num_classes, embedding_dim))
        nn.init.kaiming_normal_(self.proxies, mode="fan_out")

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = features.device
        batch_size = features.size(0)
        P = F.normalize(self.proxies, p=2, dim=-1)
        cos = F.linear(features, P)  # (B, num_classes)

        # One-hot mask for valid labels (>= 0)
        valid_mask = (labels >= 0)
        one_hot = torch.zeros(batch_size, self.num_classes, device=device)
        if valid_mask.any():
            valid_labels = labels[valid_mask].view(-1, 1)
            one_hot[valid_mask] = torch.scatter(
                torch.zeros(valid_mask.sum().item(), self.num_classes, device=device),
                1,
                valid_labels,
                1.0
            )

        has_pos = (one_hot.sum(dim=0) > 0)
        if has_pos.any():
            pos_term = torch.exp(-self.alpha * (cos - self.margin)) * one_hot
            pos_loss = torch.log(1.0 + pos_term.sum(dim=0))
            pos_loss = pos_loss[has_pos].mean()
        else:
            pos_loss = torch.tensor(0.0, device=device)

        neg_term = torch.exp(self.alpha * (cos + self.margin)) * (1.0 - one_hot)
        neg_loss = torch.log(1.0 + neg_term.sum(dim=0)).mean()

        return pos_loss + neg_loss


def build_train_only_labels(train_scaffolds, min_count: int = 3):
    counts = Counter(train_scaffolds)
    frequent = sorted([k for k, v in counts.items() if v >= min_count and k != "__EMPTY__"])
    class_to_id = {k: i for i, k in enumerate(frequent)}
    
    # Acyclic size bins for empty scaffolds in train
    empty_indices = [i for i, s in enumerate(train_scaffolds) if s == "__EMPTY__"]
    # We can assign 3 acyclic bins if min_count <= 3
    num_frequent = len(class_to_id)
    acyclic_offset = num_frequent
    
    labels = np.full(len(train_scaffolds), -1, dtype=np.int64)
    for i, s in enumerate(train_scaffolds):
        if s in class_to_id:
            labels[i] = class_to_id[s]
        elif s == "__EMPTY__":
            # 3 bins based on index or pseudo bin
            labels[i] = acyclic_offset + (i % 3)
            
    total_classes = num_frequent + 3
    return labels, class_to_id, total_classes


def train_proxy_anchor_strategy(train_ds, train_labels, num_classes: int, device: torch.device, epochs: int = 20, seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Wrap dataset with labels
    class TempDataset(torch.utils.data.Dataset):
        def __init__(self, ds, lbls):
            self.ds = ds
            self.lbls = lbls
        def __len__(self):
            return len(self.ds)
        def __getitem__(self, idx):
            d = self.ds[idx]
            d.label = torch.tensor(self.lbls[idx], dtype=torch.long)
            return d

    labeled_train = TempDataset(train_ds, train_labels)
    loader = DataLoader(labeled_train, batch_size=128, shuffle=True, drop_last=True)
    
    model = GINEncoder(num_atom_types=28, num_bond_types=5, hidden_dim=64, proj_dim=256, num_layers=4).to(device)
    criterion = ProxyAnchorWithUnlabeled(num_classes=num_classes, embedding_dim=256, margin=0.1, alpha=32.0).to(device)
    
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(criterion.parameters()), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    model.train()
    for ep in range(1, epochs + 1):
        tot_loss = 0.0
        n_b = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            z = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(z, batch.label)
            if not torch.isnan(loss) and loss.item() > 0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                tot_loss += loss.item()
                n_b += 1
        scheduler.step()
        if ep % 5 == 0 or ep == epochs:
            print(f"  Epoch [{ep:02d}/{epochs:02d}] - Loss: {tot_loss/max(n_b,1):.4f}")
            
    return model


def extract_embeddings_all(model, train_ds, val_ds, test_ds, device):
    model.eval()
    def get_embs(ds):
        loader = DataLoader(ds, batch_size=256, shuffle=False)
        out = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                z = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                out.append(z.cpu().numpy())
        return np.vstack(out)
        
    c_train = get_embs(train_ds)
    c_val = get_embs(val_ds)
    c_embs = np.vstack([c_train, c_val])
    q_embs = get_embs(test_ds)
    return c_embs, q_embs


def evaluate_embeddings(c_embs, q_embs, c_scaff, q_scaff):
    import faiss
    d = c_embs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(c_embs)
    
    # Ground truth
    scaff_to_corpus = {}
    for i, s in enumerate(c_scaff):
        if s != "__EMPTY__":
            if s not in scaff_to_corpus:
                scaff_to_corpus[s] = set()
            scaff_to_corpus[s].add(i)

    def get_gt(q_idx):
        s = q_scaff[q_idx]
        if s == "__EMPTY__":
            return set()
        return scaff_to_corpus.get(s, set())

    covered_indices = [q for q in range(len(q_scaff)) if len(get_gt(q)) > 0]
    
    # Query search
    D, I = index.search(q_embs, 50)
    
    cov_maps, cov_r10, cov_r50, cov_mrrs, cov_acc = [], [], [], [], []
    all_maps, all_r10, all_r50, all_mrrs, all_acc = [], [], [], [], []
    
    for q_idx in range(len(q_scaff)):
        gt = get_gt(q_idx)
        top50 = I[q_idx].tolist()
        m = compute_metrics_for_gt(top50, gt, c_scaff, q_scaff[q_idx])
        
        all_maps.append(m["map_10"])
        all_r10.append(m["rec_10"])
        all_r50.append(m["rec_50"])
        all_mrrs.append(m["mrr"])
        all_acc.append(m["acc_10"])
        
        if q_idx in covered_indices:
            cov_maps.append(m["map_10"])
            cov_r10.append(m["rec_10"])
            cov_r50.append(m["rec_50"])
            cov_mrrs.append(m["mrr"])
            cov_acc.append(m["acc_10"])
            
    return {
        "covered": {
            "map_10": float(np.mean(cov_maps)),
            "rec_10": float(np.mean(cov_r10)),
            "rec_50": float(np.mean(cov_r50)),
            "mrr": float(np.mean(cov_mrrs)),
            "acc_10": float(np.mean(cov_acc)),
        },
        "all_1k": {
            "map_10": float(np.mean(all_maps)),
            "rec_10": float(np.mean(all_r10)),
            "rec_50": float(np.mean(all_r50)),
            "mrr": float(np.mean(all_mrrs)),
            "acc_10": float(np.mean(all_acc)),
        }
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Training Strategy Comparison on {device}...")
    
    train_ds = ZINC(root="data/ZINC", subset=True, split="train")
    val_ds = ZINC(root="data/ZINC", subset=True, split="val")
    test_ds = ZINC(root="data/ZINC", subset=True, split="test")
    
    with open("data/exact_scaffolds.pkl", "rb") as f:
        scaff_data = pickle.load(f)
    c_scaff = scaff_data["corpus_scaffolds"]
    q_scaff = scaff_data["query_scaffolds"]
    train_scaff = c_scaff[:10000]
    
    results = {}
    
    # -------------------------------------------------------------
    # Strategy A: Legacy (evaluated from existing trained checkpoint)
    # -------------------------------------------------------------
    print("\nEvaluating Strategy A (Legacy 336-class with 30 pseudo-clusters)...")
    c_embs_a = np.load("artifacts/embs_proxyanchor_corpus.npy")
    q_embs_a = np.load("artifacts/embs_proxyanchor_query.npy")
    results["Strategy A (Legacy Leaked 336-class + Pseudo-Clusters)"] = evaluate_embeddings(c_embs_a, q_embs_a, c_scaff, q_scaff)
    
    # -------------------------------------------------------------
    # Strategy B: Train-Only Frequent Scaffolds (>= 3 in train, 244 classes)
    # Rare molecules have label -1 (no false positive grouping)
    # -------------------------------------------------------------
    print("\nTraining Strategy B: Train-Only Vocabulary (244 classes, rare unclustered)...")
    lbls_b, class_map_b, num_classes_b = build_train_only_labels(train_scaff, min_count=3)
    print(f"Strategy B: {num_classes_b} classes. Unlabeled rare samples in train: {(lbls_b == -1).sum()}")
    model_b = train_proxy_anchor_strategy(train_ds, lbls_b, num_classes_b, device, epochs=20, seed=42)
    c_embs_b, q_embs_b = extract_embeddings_all(model_b, train_ds, val_ds, test_ds, device)
    results["Strategy B (Train-Only 244-class, Rare Unclustered)"] = evaluate_embeddings(c_embs_b, q_embs_b, c_scaff, q_scaff)
    
    # -------------------------------------------------------------
    # Strategy C: Train-Only Rare-Class Strategy (>= 2 in train, 610 classes)
    # Doubletons have their own proxy, singletons unclustered (-1)
    # -------------------------------------------------------------
    print("\nTraining Strategy C: Train-Only Expanded Vocabulary (610 classes, doubletons included, singletons unclustered)...")
    lbls_c, class_map_c, num_classes_c = build_train_only_labels(train_scaff, min_count=2)
    print(f"Strategy C: {num_classes_c} classes. Unlabeled rare samples in train: {(lbls_c == -1).sum()}")
    model_c = train_proxy_anchor_strategy(train_ds, lbls_c, num_classes_c, device, epochs=20, seed=42)
    c_embs_c, q_embs_c = extract_embeddings_all(model_c, train_ds, val_ds, test_ds, device)
    results["Strategy C (Train-Only 610-class, Doubletons Included, Rare Unclustered)"] = evaluate_embeddings(c_embs_c, q_embs_c, c_scaff, q_scaff)
    
    # Print summary table
    print("\n" + "="*85)
    print(f"{'Training Strategy':<45} | {'Cov MAP@10':<11} | {'Cov R@10':<10} | {'Cov MRR':<10} | {'Cov Acc@10':<10}")
    print("="*85)
    for strat, res in results.items():
        cov = res["covered"]
        print(f"{strat:<45} | {cov['map_10']:<11.4f} | {cov['rec_10']:<10.4f} | {cov['mrr']:<10.4f} | {cov['acc_10']:<10.4f}")
    print("="*85)
    
    out_file = "artifacts/training_strategies_comparison.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved comparison to {out_file}")

if __name__ == "__main__":
    main()
