"""
train_strategy_b_objectives.py - Train and Evaluate All GIN Objectives under Leak-Free Strategy B Taxonomy:
  1. GIN-CE (Cross-Entropy, 244 classes)
  2. GIN-SupCon (Supervised Contrastive, 244 classes, Balanced Sampling)
  3. GIN-SupCon-Aug (SupCon + GraphCL Augmentations)
  4. GIN-ProxyAnchor (Proxy-Anchor Metric Learning, 244 Proxies)
  5. GIN-ProxyAnchor (Proxy-Anchor, Label -1 Excluded)

Evaluates exact canonical Murcko scaffold retrieval for BOTH:
  - FAISS IndexFlatIP (Exact Cosine)
  - FAISS HNSW (Approximate, efSearch=64)
Calculates exact ANN Recall@10 against Flat IP.
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
import faiss

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.gin_encoder import (
    GINEncoder, GINClassifier, SupConLoss, drop_edges, mask_features
)
from src.data_ingestion import ClassBalancedSampler


# -------------------------------------------------------------
# 1. Custom Dataset & Proxy-Anchor Variations
# -------------------------------------------------------------
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


class ProxyAnchorWithUnlabeled(nn.Module):
    """
    Proxy-Anchor loss where samples with label == -1 act as negative repellers
    against all class proxies, pulling no positive proxy.
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


class ProxyAnchorExcludedUnlabeled(nn.Module):
    """
    Proxy-Anchor loss where samples with label == -1 are completely excluded
    from both positive and negative loss terms (no false-negative repulsion).
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
        valid_mask = (labels >= 0)
        if not valid_mask.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        features = features[valid_mask]
        labels = labels[valid_mask]
        batch_size = features.size(0)

        P = F.normalize(self.proxies, p=2, dim=-1)
        cos = F.linear(features, P)

        one_hot = torch.zeros(batch_size, self.num_classes, device=device)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

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


# -------------------------------------------------------------
# 2. Strategy B Taxonomy Builder
# -------------------------------------------------------------
def build_strategy_b_taxonomy(train_scaffolds):
    """
    Construct vocabulary STRICTLY from 10,000 training graphs.
    """
    counts = Counter(train_scaffolds)
    frequent = sorted([k for k, v in counts.items() if v >= 3 and k != "__EMPTY__"])
    class_to_id = {k: i for i, k in enumerate(frequent)}
    num_frequent = len(class_to_id)
    acyclic_offset = num_frequent
    
    labels = np.full(len(train_scaffolds), -1, dtype=np.int64)
    for i, s in enumerate(train_scaffolds):
        if s in class_to_id:
            labels[i] = class_to_id[s]
        elif s == "__EMPTY__":
            labels[i] = acyclic_offset + (i % 3)
            
    total_classes = num_frequent + 3  # 241 + 3 = 244
    return labels, class_to_id, total_classes


# -------------------------------------------------------------
# 3. Model Training Functions
# -------------------------------------------------------------
def train_ce_strategy_b(train_ds, train_labels, num_classes, device, epochs=20, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Filter training to valid classes (labels >= 0)
    valid_indices = [i for i, lbl in enumerate(train_labels) if lbl >= 0]
    valid_subset = torch.utils.data.Subset(train_ds, valid_indices)
    valid_labels = [train_labels[i] for i in valid_indices]
    labeled_train = LabeledGraphDataset(valid_subset, valid_labels)

    loader = DataLoader(labeled_train, batch_size=128, shuffle=True, drop_last=True)
    encoder = GINEncoder(num_atom_types=28, num_bond_types=5, hidden_dim=64, proj_dim=256, num_layers=4).to(device)
    model = GINClassifier(encoder, num_classes=num_classes, proj_dim=256).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    for ep in range(1, epochs + 1):
        tot_loss = 0.0
        n_b = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(logits, batch.label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tot_loss += loss.item()
            n_b += 1
        scheduler.step()
        if ep % 5 == 0 or ep == epochs:
            print(f"  [GIN-CE Strategy B] Ep [{ep:02d}/{epochs:02d}] - Loss: {tot_loss/max(n_b,1):.4f}")
    return model.encoder


def train_supcon_strategy_b(train_ds, train_labels, device, epochs=20, seed=42, use_augmentations=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    name = "GIN-SupCon-Aug" if use_augmentations else "GIN-SupCon"

    valid_indices = [i for i, lbl in enumerate(train_labels) if lbl >= 0]
    valid_subset = torch.utils.data.Subset(train_ds, valid_indices)
    valid_labels = np.array([train_labels[i] for i in valid_indices])
    labeled_train = LabeledGraphDataset(valid_subset, valid_labels)

    # Class-balanced sampler over valid classes
    sampler = ClassBalancedSampler(valid_labels, p_classes=16, k_instances=8, batches_per_epoch=75)
    loader = DataLoader(labeled_train, batch_sampler=sampler)

    model = GINEncoder(num_atom_types=28, num_bond_types=5, hidden_dim=64, proj_dim=256, num_layers=4).to(device)
    criterion = SupConLoss(temperature=0.07).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    for ep in range(1, epochs + 1):
        tot_loss = 0.0
        n_b = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            if use_augmentations:
                aug_edge_index, aug_edge_attr = drop_edges(batch.edge_index, batch.edge_attr, p=0.1)
                aug_x = mask_features(batch.x, p=0.1)
                z = model(aug_x, aug_edge_index, aug_edge_attr, batch.batch)
            else:
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
            print(f"  [{name} Strategy B] Ep [{ep:02d}/{epochs:02d}] - Loss: {tot_loss/max(n_b,1):.4f}")
    return model


def train_proxy_anchor_variant(train_ds, train_labels, num_classes, criterion_cls, device, epochs=20, seed=42, desc=""):
    torch.manual_seed(seed)
    np.random.seed(seed)

    labeled_train = LabeledGraphDataset(train_ds, train_labels)
    loader = DataLoader(labeled_train, batch_size=128, shuffle=True, drop_last=True)

    model = GINEncoder(num_atom_types=28, num_bond_types=5, hidden_dim=64, proj_dim=256, num_layers=4).to(device)
    criterion = criterion_cls(num_classes=num_classes, embedding_dim=256, margin=0.1, alpha=32.0).to(device)

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
            print(f"  [{desc}] Ep [{ep:02d}/{epochs:02d}] - Loss: {tot_loss/max(n_b,1):.4f}")
    return model


# -------------------------------------------------------------
# 4. Evaluation Engine (Exact Murcko Relevance + Flat IP + HNSW)
# -------------------------------------------------------------
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
    c_embs = np.vstack([c_train, c_val]).astype(np.float32)
    q_embs = get_embs(test_ds).astype(np.float32)
    return c_embs, q_embs


def compute_metrics_for_ranking(top_indices, gt, c_scaff, q_scaff):
    res = {}
    if not gt:
        res["map_10"] = 0.0
        res["rec_10"] = 0.0
        res["rec_50"] = 0.0
        res["mrr"] = 0.0
        res["acc_10"] = 0.0
        return res

    hits = [1 if idx in gt else 0 for idx in top_indices[:50]]
    
    # MAP@10 and Recall@10
    num_hits = 0
    prec_sum = 0.0
    for rank, hit in enumerate(hits[:10], 1):
        if hit:
            num_hits += 1
            prec_sum += (num_hits / rank)
    denom = min(10, len(gt))
    res["map_10"] = (prec_sum / denom) if denom > 0 else 0.0
    res["rec_10"] = num_hits / len(gt)

    # Recall@50
    res["rec_50"] = sum(hits[:50]) / len(gt)

    # MRR
    rr = 0.0
    for rank, hit in enumerate(hits, 1):
        if hit:
            rr = 1.0 / rank
            break
    res["mrr"] = rr

    # Majority-vote Acc@10 with explicit deterministic rank tie-breaker
    top_indices = list(top_indices)
    top10 = top_indices[:10]
    if len(top10) > 0:
        retrieved_scaffs = [c_scaff[idx] for idx in top10]
        counts = Counter(retrieved_scaffs)
        max_c = max(counts.values())
        top_candidates = {s for s, c in counts.items() if c == max_c}
        mode_scaff = None
        for s in retrieved_scaffs:
            if s in top_candidates:
                mode_scaff = s
                break
        res["acc_10"] = 1.0 if mode_scaff == q_scaff else 0.0
    else:
        res["acc_10"] = 0.0

    return res


def evaluate_index_system(I, c_scaff, q_scaff, covered_indices):
    cov_maps, cov_r10, cov_r50, cov_mrrs, cov_acc = [], [], [], [], []
    all_maps, all_r10, all_r50, all_mrrs, all_acc = [], [], [], [], []

    # Map scaffold to corpus ground truth
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

    for q in range(len(q_scaff)):
        gt = get_gt(q)
        m = compute_metrics_for_ranking(I[q], gt, c_scaff, q_scaff[q])
        all_maps.append(m["map_10"])
        all_r10.append(m["rec_10"])
        all_r50.append(m["rec_50"])
        all_mrrs.append(m["mrr"])
        all_acc.append(m["acc_10"])

        if q in covered_indices:
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
    print(f"=== Running Strategy B Unified Benchmark on {device} ===")

    train_ds = ZINC(root="data/ZINC", subset=True, split="train")
    val_ds = ZINC(root="data/ZINC", subset=True, split="val")
    test_ds = ZINC(root="data/ZINC", subset=True, split="test")

    with open("data/exact_scaffolds.pkl", "rb") as f:
        scaff_data = pickle.load(f)
    c_scaff = scaff_data["corpus_scaffolds"]
    q_scaff = scaff_data["query_scaffolds"]
    train_scaff = c_scaff[:10000]

    # Build Ground Truth Mapping
    scaff_to_corpus = {}
    for i, s in enumerate(c_scaff):
        if s != "__EMPTY__":
            if s not in scaff_to_corpus:
                scaff_to_corpus[s] = set()
            scaff_to_corpus[s].add(i)

    covered_indices = [q for q in range(len(q_scaff)) if q_scaff[q] != "__EMPTY__" and len(scaff_to_corpus.get(q_scaff[q], set())) > 0]
    print(f"Total queries: {len(q_scaff)}, Covered: {len(covered_indices)} ({len(covered_indices)/len(q_scaff)*100:.2f}%)")

    # Strategy B Taxonomy
    train_labels_b, class_map_b, num_classes_b = build_strategy_b_taxonomy(train_scaff)
    print(f"Strategy B Taxonomy: {num_classes_b} classes derived strictly from train data.")
    print(f"  Molecules in train with class >= 0: {(train_labels_b >= 0).sum()}")
    print(f"  Molecules in train with label -1:   {(train_labels_b == -1).sum()}")

    models = {}

    # 1. GIN-CE Strategy B
    print("\n--- Training Model 1: GIN-CE (Strategy B) ---")
    m_ce = train_ce_strategy_b(train_ds, train_labels_b, num_classes_b, device, epochs=20, seed=42)
    models["GIN-CE (Strategy B)"] = m_ce
    torch.save(m_ce.state_dict(), "artifacts/gin_strat_b_ce.pt")

    # 2. GIN-SupCon Strategy B
    print("\n--- Training Model 2: GIN-SupCon (Strategy B) ---")
    m_supcon = train_supcon_strategy_b(train_ds, train_labels_b, device, epochs=20, seed=42, use_augmentations=False)
    models["GIN-SupCon (Strategy B)"] = m_supcon
    torch.save(m_supcon.state_dict(), "artifacts/gin_strat_b_supcon.pt")

    # 3. GIN-SupCon-Aug Strategy B
    print("\n--- Training Model 3: GIN-SupCon-Aug (Strategy B) ---")
    m_supcon_aug = train_supcon_strategy_b(train_ds, train_labels_b, device, epochs=20, seed=42, use_augmentations=True)
    models["GIN-SupCon-Aug (Strategy B)"] = m_supcon_aug
    torch.save(m_supcon_aug.state_dict(), "artifacts/gin_strat_b_supcon_aug.pt")

    # 4. GIN-ProxyAnchor Strategy B (Label -1 as negative repeller)
    print("\n--- Training Model 4: GIN-ProxyAnchor (Strategy B, Label -1 Negative) ---")
    m_pa_neg = train_proxy_anchor_variant(train_ds, train_labels_b, num_classes_b, ProxyAnchorWithUnlabeled, device, epochs=20, seed=42, desc="GIN-PA-Neg")
    models["GIN-ProxyAnchor (Strategy B, Neg-Repeller)"] = m_pa_neg
    torch.save(m_pa_neg.state_dict(), "artifacts/gin_strat_b_proxyanchor_neg.pt")

    # 5. GIN-ProxyAnchor Strategy B (Label -1 Excluded)
    print("\n--- Training Model 5: GIN-ProxyAnchor (Strategy B, Label -1 Excluded) ---")
    m_pa_excl = train_proxy_anchor_variant(train_ds, train_labels_b, num_classes_b, ProxyAnchorExcludedUnlabeled, device, epochs=20, seed=42, desc="GIN-PA-Excl")
    models["GIN-ProxyAnchor (Strategy B, Excluded)"] = m_pa_excl
    torch.save(m_pa_excl.state_dict(), "artifacts/gin_strat_b_proxyanchor_excl.pt")

    # -------------------------------------------------------------
    # 5. Extract Embeddings and Evaluate Both Flat IP and HNSW
    # -------------------------------------------------------------
    benchmark_results = {}

    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        c_embs, q_embs = extract_embeddings_all(model, train_ds, val_ds, test_ds, device)

        # 1. Exact Flat IP
        flat = faiss.IndexFlatIP(256)
        flat.add(c_embs)
        _, flat_I = flat.search(q_embs, 50)
        flat_res = evaluate_index_system(flat_I, c_scaff, q_scaff, covered_indices)
        benchmark_results[f"{name} + Flat IP (Exact)"] = flat_res

        # 2. Approximate HNSW (M=32, efConstruction=200, efSearch=64)
        hnsw = faiss.IndexHNSWFlat(256, 32, faiss.METRIC_INNER_PRODUCT)
        hnsw.hnsw.efSearch = 64
        hnsw.add(c_embs)
        _, hnsw_I = hnsw.search(q_embs, 50)
        hnsw_res = evaluate_index_system(hnsw_I, c_scaff, q_scaff, covered_indices)

        # Compute exact ANN Recall@10
        ann_rec_list = [len(set(flat_I[q, :10]) & set(hnsw_I[q, :10])) / 10.0 for q in range(len(q_embs))]
        ann_rec_10 = float(np.mean(ann_rec_list))
        hnsw_res["ann_recall_at_10"] = ann_rec_10
        benchmark_results[f"{name} + HNSW"] = hnsw_res

        cov_flat = flat_res["covered"]
        cov_hnsw = hnsw_res["covered"]
        print(f"  Exact Flat IP -> MAP@10: {cov_flat['map_10']:.4f} | R@10: {cov_flat['rec_10']:.4f} | MRR: {cov_flat['mrr']:.4f} | Acc@10: {cov_flat['acc_10']:.4f}")
        print(f"  HNSW (ef=64)  -> MAP@10: {cov_hnsw['map_10']:.4f} | R@10: {cov_hnsw['rec_10']:.4f} | MRR: {cov_hnsw['mrr']:.4f} | Acc@10: {cov_hnsw['acc_10']:.4f} | ANN-Rec@10: {ann_rec_10:.4f}")

    # Save to JSON
    out_path = "artifacts/strategy_b_objectives_comparison.json"
    with open(out_path, "w") as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"\nAll Strategy B benchmark results saved to {out_path}!")

    # Print summary table
    print("\n" + "="*95)
    print(f"{'Pipeline (Strategy B Controlled Taxonomy)':<45} | {'Cov MAP@10':<11} | {'Cov R@10':<10} | {'Cov MRR':<10} | {'Cov Acc@10':<10} | {'ANN-Rec':<8}")
    print("="*95)
    for p_name, p_data in benchmark_results.items():
        cov = p_data["covered"]
        ann = p_data.get("ann_recall_at_10", 1.0)
        print(f"{p_name:<45} | {cov['map_10']:<11.4f} | {cov['rec_10']:<10.4f} | {cov['mrr']:<10.4f} | {cov['acc_10']:<10.4f} | {ann:<8.4f}")
    print("="*95)


if __name__ == "__main__":
    main()
