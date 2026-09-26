import os
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

from scipy.stats import pearsonr, spearmanr
from transformers import AutoTokenizer, AutoModel


@dataclass
class Config:
    csv_path: str = "biomap.csv"
    heavy_col: str = "ab"
    antigen_col: str = "ag"
    label_col: str = "g"

    model_name: str = "facebook/esm2_t6_8M_UR50D"

    max_heavy_len: int = 256
    max_antigen_len: int = 512

    batch_size: int = 8
    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-5

    num_heads: int = 8
    hidden_dim: int = 256
    dropout: float = 0.1

    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    num_runs: int = 5
    base_seed: int = 42
    num_workers: int = 0

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = "affinity_five_run_results"


cfg = Config()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def masked_mean(x, mask):
    mask = mask.unsqueeze(-1).float()
    x = x * mask
    return x.sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def safe_pearson(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) < 2:
        return float("nan")
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return float("nan")
    return pearsonr(y_true, y_pred)[0]


def safe_spearman(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) < 2:
        return float("nan")
    return spearmanr(y_true, y_pred)[0]



class PairRegressionDataset(Dataset):
    def __init__(self, csv_path, heavy_col, antigen_col, label_col):
        df = pd.read_csv(csv_path)

        for col in [heavy_col, antigen_col, label_col]:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in {csv_path}")

        df = df[[heavy_col, antigen_col, label_col]].dropna().copy()

        df[heavy_col] = df[heavy_col].astype(str).str.strip().str.upper()
        df[antigen_col] = df[antigen_col].astype(str).str.strip().str.upper()
        df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
        df = df.dropna().copy()

        df = df[
            (df[heavy_col].str.len() > 0) &
            (df[antigen_col].str.len() > 0)
        ].reset_index(drop=True)

        self.samples = list(
            zip(
                df[heavy_col].tolist(),
                df[antigen_col].tolist(),
                df[label_col].astype(float).tolist()
            )
        )

        if len(self.samples) == 0:
            raise ValueError("No valid samples after filtering.")

        print(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        heavy, antigen, label = self.samples[idx]
        return {
            "heavy": heavy,
            "antigen": antigen,
            "label": float(label),
        }



class PairCollator:
    def __init__(self, tokenizer, max_heavy_len, max_antigen_len):
        self.tokenizer = tokenizer
        self.max_heavy_len = max_heavy_len
        self.max_antigen_len = max_antigen_len

    @staticmethod
    def add_spaces(seq: str) -> str:
        return " ".join(list(seq))

    def __call__(self, batch):
        heavy_texts = [self.add_spaces(item["heavy"]) for item in batch]
        antigen_texts = [self.add_spaces(item["antigen"]) for item in batch]
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.float32)

        heavy_inputs = self.tokenizer(
            heavy_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_heavy_len,
        )

        antigen_inputs = self.tokenizer(
            antigen_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_antigen_len,
        )

        return {
            "heavy_input_ids": heavy_inputs["input_ids"],
            "heavy_attention_mask": heavy_inputs["attention_mask"],
            "antigen_input_ids": antigen_inputs["input_ids"],
            "antigen_attention_mask": antigen_inputs["attention_mask"],
            "labels": labels,
        }



class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value, key_padding_mask=None):
        out, attn_weights = self.attn(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=key_padding_mask,
            need_weights=True,
        )
        out = self.norm(query + self.dropout(out))
        return out, attn_weights



class ESM2BidirectionalCrossAttentionRegressor(nn.Module):
    def __init__(self, model_name, hidden_dim=256, num_heads=8, dropout=0.1):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.esm = AutoModel.from_pretrained(model_name)

        for p in self.esm.parameters():
            p.requires_grad = False

        esm_dim = self.esm.config.hidden_size

        self.ab_to_ag = CrossAttentionBlock(
            dim=esm_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.ag_to_ab = CrossAttentionBlock(
            dim=esm_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.ab_proj = nn.Linear(esm_dim, hidden_dim)
        self.ag_proj = nn.Linear(esm_dim, hidden_dim)

        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, input_ids, attention_mask):
        outputs = self.esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return outputs.last_hidden_state

    def forward(
        self,
        heavy_input_ids,
        heavy_attention_mask,
        antigen_input_ids,
        antigen_attention_mask,
    ):
        with torch.no_grad():
            heavy_emb = self.encode(heavy_input_ids, heavy_attention_mask)
            antigen_emb = self.encode(antigen_input_ids, antigen_attention_mask)

        antigen_key_padding_mask = (antigen_attention_mask == 0)
        heavy_key_padding_mask = (heavy_attention_mask == 0)

        heavy_ctx, heavy_to_antigen_attn = self.ab_to_ag(
            query=heavy_emb,
            key_value=antigen_emb,
            key_padding_mask=antigen_key_padding_mask,
        )

        antigen_ctx, antigen_to_heavy_attn = self.ag_to_ab(
            query=antigen_emb,
            key_value=heavy_emb,
            key_padding_mask=heavy_key_padding_mask,
        )

        heavy_vec = masked_mean(self.ab_proj(heavy_ctx), heavy_attention_mask)
        antigen_vec = masked_mean(self.ag_proj(antigen_ctx), antigen_attention_mask)

        pair_feat = torch.cat([
            heavy_vec,
            antigen_vec,
            torch.abs(heavy_vec - antigen_vec),
            heavy_vec * antigen_vec,
        ], dim=-1)

        preds = self.regressor(pair_feat).squeeze(-1)
        return preds, heavy_to_antigen_attn, antigen_to_heavy_attn

def compute_regression_metrics(labels, preds):
    labels = np.asarray(labels, dtype=float)
    preds = np.asarray(preds, dtype=float)

    return {
        "Pearson": safe_pearson(labels, preds),
        "Spearman": safe_spearman(labels, preds),
    }



def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    # Keep the frozen ESM-2 encoder deterministic during training.
    model.esm.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in loader:
        heavy_input_ids = batch["heavy_input_ids"].to(device)
        heavy_attention_mask = batch["heavy_attention_mask"].to(device)
        antigen_input_ids = batch["antigen_input_ids"].to(device)
        antigen_attention_mask = batch["antigen_attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        preds, _, _ = model(
            heavy_input_ids=heavy_input_ids,
            heavy_attention_mask=heavy_attention_mask,
            antigen_input_ids=antigen_input_ids,
            antigen_attention_mask=antigen_attention_mask,
        )

        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())

    metrics = compute_regression_metrics(all_labels, all_preds)
    return total_loss / len(loader), metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in loader:
        heavy_input_ids = batch["heavy_input_ids"].to(device)
        heavy_attention_mask = batch["heavy_attention_mask"].to(device)
        antigen_input_ids = batch["antigen_input_ids"].to(device)
        antigen_attention_mask = batch["antigen_attention_mask"].to(device)
        labels = batch["labels"].to(device)

        preds, _, _ = model(
            heavy_input_ids=heavy_input_ids,
            heavy_attention_mask=heavy_attention_mask,
            antigen_input_ids=antigen_input_ids,
            antigen_attention_mask=antigen_attention_mask,
        )

        loss = criterion(preds, labels)
        total_loss += loss.item()
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())

    metrics = compute_regression_metrics(all_labels, all_preds)
    return total_loss / len(loader), metrics



def make_split_indices(n_samples, seed):
    if not np.isclose(cfg.train_ratio + cfg.val_ratio + cfg.test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    indices = np.arange(n_samples)
    train_val_idx, test_idx = train_test_split(
        indices, test_size=cfg.test_ratio, random_state=seed, shuffle=True
    )
    val_fraction = cfg.val_ratio / (cfg.train_ratio + cfg.val_ratio)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_fraction, random_state=seed, shuffle=True
    )
    return np.asarray(train_idx), np.asarray(val_idx), np.asarray(test_idx)

def make_loader(dataset, indices, collator, shuffle, seed):
    g = torch.Generator().manual_seed(seed)
    return DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=cfg.batch_size, shuffle=shuffle,
        num_workers=cfg.num_workers, collate_fn=collator,
        generator=g if shuffle else None,
    )

def main():
    os.makedirs(cfg.output_dir, exist_ok=True)
    split_dir = os.path.join(cfg.output_dir, "splits")
    ckpt_dir = os.path.join(cfg.output_dir, "checkpoints")
    os.makedirs(split_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    dataset = PairRegressionDataset(
        csv_path=cfg.csv_path, heavy_col=cfg.heavy_col,
        antigen_col=cfg.antigen_col, label_col=cfg.label_col,
    )
    collator = PairCollator(tokenizer, cfg.max_heavy_len, cfg.max_antigen_len)
    criterion = nn.MSELoss()
    rows = []

    for run in range(1, cfg.num_runs + 1):
        seed = cfg.base_seed + run - 1
        set_seed(seed)
        train_idx, val_idx, test_idx = make_split_indices(len(dataset), seed)

      
        assert not (set(train_idx) & set(val_idx))
        assert not (set(train_idx) & set(test_idx))
        assert not (set(val_idx) & set(test_idx))
        assert len(set(train_idx) | set(val_idx) | set(test_idx)) == len(dataset)

        np.savez(
            os.path.join(split_dir, f"split_run{run}.npz"),
            train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, seed=seed,
        )
        print(f"\
Run {run}/5 seed={seed}: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

        train_loader = make_loader(dataset, train_idx, collator, True, seed)
        val_loader = make_loader(dataset, val_idx, collator, False, seed)
        test_loader = make_loader(dataset, test_idx, collator, False, seed)

        set_seed(seed)
        model = ESM2BidirectionalCrossAttentionRegressor(
            model_name=cfg.model_name, hidden_dim=cfg.hidden_dim,
            num_heads=cfg.num_heads, dropout=cfg.dropout,
        ).to(cfg.device)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.lr, weight_decay=cfg.weight_decay,
        )

        best_val_pearson = -float("inf")
        best_epoch = -1
        ckpt = os.path.join(ckpt_dir, f"best_run{run}_seed{seed}.pt")

        for epoch in range(1, cfg.epochs + 1):
            train_loss, train_metrics = train_one_epoch(
                model, train_loader, optimizer, criterion, cfg.device
            )
            val_loss, val_metrics = evaluate(model, val_loader, criterion, cfg.device)
            print(
                f"Epoch {epoch:02d} | train_loss={train_loss:.4f} "
                f"train_Pearson={train_metrics['Pearson']:.4f} "
                f"train_Spearman={train_metrics['Spearman']:.4f} | "
                f"val_loss={val_loss:.4f} val_Pearson={val_metrics['Pearson']:.4f} "
                f"val_Spearman={val_metrics['Spearman']:.4f}"
            )
            vp = val_metrics["Pearson"]
            if not np.isnan(vp) and vp > best_val_pearson:
                best_val_pearson = float(vp)
                best_epoch = epoch
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "run": run, "seed": seed, "best_epoch": epoch,
                    "best_val_pearson": best_val_pearson,
                }, ckpt)

        if best_epoch < 0:
            raise RuntimeError(f"Run {run}: no valid validation Pearson obtained")

       
        state = torch.load(ckpt, map_location=cfg.device)
        model.load_state_dict(state["model_state_dict"])
        test_loss, test_metrics = evaluate(model, test_loader, criterion, cfg.device)
        print(
            f"TEST run={run}: Pearson={test_metrics['Pearson']:.4f}, "
            f"Spearman={test_metrics['Spearman']:.4f}, MSE={test_loss:.4f}"
        )
        rows.append({
            "run": run, "seed": seed, "best_epoch": best_epoch,
            "best_val_Pearson": best_val_pearson, "test_MSE": test_loss,
            "test_Pearson": test_metrics["Pearson"],
            "test_Spearman": test_metrics["Spearman"],
        })

        del model, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results = pd.DataFrame(rows)
    results.to_csv(os.path.join(cfg.output_dir, "five_run_results.csv"), index=False)

    summary = []
    for metric in ["test_MSE", "test_Pearson", "test_Spearman"]:
        vals = results[metric].astype(float)
        summary.append({
            "metric": metric, "n_runs": len(vals),
            "mean": vals.mean(), "sd": vals.std(ddof=1),
            "mean_plus_minus_sd": f"{vals.mean():.4f} ± {vals.std(ddof=1):.4f}",
        })
    summary = pd.DataFrame(summary)
    summary.to_csv(os.path.join(cfg.output_dir, "five_run_summary.csv"), index=False)

    print("\
Five-run held-out test results:")
    print(results.to_string(index=False))
    print("\
Mean ± s.d. (sample SD, ddof=1):")
    print(summary[["metric", "mean_plus_minus_sd"]].to_string(index=False))

if __name__ == "__main__":
    main()
