import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    matthews_corrcoef,
    accuracy_score,
)

from transformers import AutoTokenizer, AutoModel



@dataclass
class Config:
    csv_path: str = "CoV-AbDab.csv"
    heavy_col: str = "Heavy"
    antigen_col: str = "antigen"
    label_col: str = "Label"

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

    val_ratio: float = 0.1
    seed: int = 42
    num_workers: int = 0

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_path: str = "best_esm2_cross_attention.pt"


cfg = Config()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def masked_mean(x, mask):
    # x: [B, L, D]
    # mask: [B, L], 1 valid / 0 pad
    mask = mask.unsqueeze(-1).float()
    x = x * mask
    return x.sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def safe_auc(labels, probs):
    try:
        return roc_auc_score(labels, probs)
    except ValueError:
        return float("nan")

# Dataset

class PairDataset(Dataset):
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
        df[label_col] = df[label_col].astype(int)

        df = df[
            (df[heavy_col].str.len() > 0) &
            (df[antigen_col].str.len() > 0)
        ].reset_index(drop=True)

        self.samples = list(
            zip(
                df[heavy_col].tolist(),
                df[antigen_col].tolist(),
                df[label_col].tolist()
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
            key_padding_mask=key_padding_mask,   # True means ignore
            need_weights=True,
        )
        out = self.norm(query + self.dropout(out))
        return out, attn_weights


class ESM2BidirectionalCrossAttentionClassifier(nn.Module):
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

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def get_input_embeddings(self):
        return self.esm.get_input_embeddings()

    def encode_from_ids(self, input_ids, attention_mask):
        outputs = self.esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return outputs.last_hidden_state  # [B, L, D]

    def encode_from_embeds(self, inputs_embeds, attention_mask):
        outputs = self.esm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        return outputs.last_hidden_state  # [B, L, D]

    def _forward_from_hidden(
        self,
        heavy_emb,
        heavy_attention_mask,
        antigen_emb,
        antigen_attention_mask,
    ):
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

        logits = self.classifier(pair_feat).squeeze(-1)
        return logits, heavy_to_antigen_attn, antigen_to_heavy_attn

    def forward(
        self,
        heavy_input_ids,
        heavy_attention_mask,
        antigen_input_ids,
        antigen_attention_mask,
    ):
        with torch.no_grad():
            heavy_emb = self.encode_from_ids(heavy_input_ids, heavy_attention_mask)
            antigen_emb = self.encode_from_ids(antigen_input_ids, antigen_attention_mask)

        return self._forward_from_hidden(
            heavy_emb=heavy_emb,
            heavy_attention_mask=heavy_attention_mask,
            antigen_emb=antigen_emb,
            antigen_attention_mask=antigen_attention_mask,
        )

    def forward_from_embeds(
        self,
        heavy_inputs_embeds,
        heavy_attention_mask,
        antigen_inputs_embeds,
        antigen_attention_mask,
    ):
       
        heavy_emb = self.encode_from_embeds(heavy_inputs_embeds, heavy_attention_mask)
        antigen_emb = self.encode_from_embeds(antigen_inputs_embeds, antigen_attention_mask)

        return self._forward_from_hidden(
            heavy_emb=heavy_emb,
            heavy_attention_mask=heavy_attention_mask,
            antigen_emb=antigen_emb,
            antigen_attention_mask=antigen_attention_mask,
        )


def compute_metrics(labels, probs, threshold=0.5):
    labels = np.array(labels).astype(int)
    probs = np.array(probs)
    preds = (probs >= threshold).astype(int)

    metrics = {}
    metrics["AUC"] = safe_auc(labels, probs)
    metrics["F1"] = f1_score(labels, preds, zero_division=0)
    metrics["MCC"] = matthews_corrcoef(labels, preds)
    metrics["Accuracy"] = accuracy_score(labels, preds)

    return metrics


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    all_probs = []
    all_labels = []

    for batch in loader:
        heavy_input_ids = batch["heavy_input_ids"].to(device)
        heavy_attention_mask = batch["heavy_attention_mask"].to(device)
        antigen_input_ids = batch["antigen_input_ids"].to(device)
        antigen_attention_mask = batch["antigen_attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        logits, _, _ = model(
            heavy_input_ids=heavy_input_ids,
            heavy_attention_mask=heavy_attention_mask,
            antigen_input_ids=antigen_input_ids,
            antigen_attention_mask=antigen_attention_mask,
        )

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        probs = torch.sigmoid(logits)
        all_probs.extend(probs.detach().cpu().numpy().tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())

    metrics = compute_metrics(all_labels, all_probs)
    return total_loss / len(loader), metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    all_probs = []
    all_labels = []

    for batch in loader:
        heavy_input_ids = batch["heavy_input_ids"].to(device)
        heavy_attention_mask = batch["heavy_attention_mask"].to(device)
        antigen_input_ids = batch["antigen_input_ids"].to(device)
        antigen_attention_mask = batch["antigen_attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits, _, _ = model(
            heavy_input_ids=heavy_input_ids,
            heavy_attention_mask=heavy_attention_mask,
            antigen_input_ids=antigen_input_ids,
            antigen_attention_mask=antigen_attention_mask,
        )

        loss = criterion(logits, labels)
        total_loss += loss.item()

        probs = torch.sigmoid(logits)
        all_probs.extend(probs.detach().cpu().numpy().tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())

    metrics = compute_metrics(all_labels, all_probs)
    return total_loss / len(loader), metrics


def add_spaces(seq: str) -> str:
    return " ".join(list(seq.strip().upper()))


def tokenize_single_sequence(tokenizer, seq: str, max_length: int):
    text = add_spaces(seq)
    enc = tokenizer(
        text,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=max_length,
        return_special_tokens_mask=True,
    )
    return enc


def extract_residue_tokens_and_scores(
    tokenizer,
    input_ids: torch.Tensor,            # [L]
    attention_mask: torch.Tensor,       # [L]
    scores: torch.Tensor,               # [L]
    original_seq: str,
) -> List[Dict]:

    token_ids = input_ids.detach().cpu().tolist()
    attn = attention_mask.detach().cpu().tolist()
    score_vals = scores.detach().cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    results = []
    residue_idx = 0

    for tok, mask_val, score in zip(tokens, attn, score_vals):
        if mask_val == 0:
            continue

        if tok in tokenizer.all_special_tokens:
            continue

        residue_char = original_seq[residue_idx] if residue_idx < len(original_seq) else tok
        results.append({
            "position_1based": residue_idx + 1,
            "residue": residue_char,
            "token": tok,
            "importance": float(score),
        })
        residue_idx += 1

    return results


def normalize_scores(scores: torch.Tensor) -> torch.Tensor:
    if scores.numel() == 0:
        return scores
    smin = scores.min()
    smax = scores.max()
    if float(smax - smin) < 1e-12:
        return torch.zeros_like(scores)
    return (scores - smin) / (smax - smin)


@torch.no_grad()
def predict_single(
    model,
    tokenizer,
    heavy_seq: str,
    antigen_seq: str,
    device: str,
    max_heavy_len: int,
    max_antigen_len: int,
) -> Dict:
    model.eval()

    heavy_enc = tokenize_single_sequence(tokenizer, heavy_seq, max_heavy_len)
    antigen_enc = tokenize_single_sequence(tokenizer, antigen_seq, max_antigen_len)

    heavy_input_ids = heavy_enc["input_ids"].to(device)
    heavy_attention_mask = heavy_enc["attention_mask"].to(device)

    antigen_input_ids = antigen_enc["input_ids"].to(device)
    antigen_attention_mask = antigen_enc["attention_mask"].to(device)

    logits, h2a_attn, a2h_attn = model(
        heavy_input_ids=heavy_input_ids,
        heavy_attention_mask=heavy_attention_mask,
        antigen_input_ids=antigen_input_ids,
        antigen_attention_mask=antigen_attention_mask,
    )

    prob = torch.sigmoid(logits)[0].item()

    return {
        "logit": logits[0].item(),
        "prob": prob,
        "heavy_to_antigen_attn": h2a_attn,
        "antigen_to_heavy_attn": a2h_attn,
    }


def attribute_positive_probability_to_input_embeddings(
    model,
    tokenizer,
    heavy_seq: str,
    antigen_seq: str,
    device: str,
    max_heavy_len: int,
    max_antigen_len: int,
    score_mode: str = "grad_x_input",   # "grad_norm" or "grad_x_input"
    normalize: bool = True,
) -> Dict:
    """
    score_mode:
        - grad_norm: || d p / d e_i ||
        - grad_x_input: sum_j | (d p / d e_ij) * e_ij |
    """
    model.eval()

    heavy_seq = heavy_seq.strip().upper()
    antigen_seq = antigen_seq.strip().upper()

    heavy_enc = tokenize_single_sequence(tokenizer, heavy_seq, max_heavy_len)
    antigen_enc = tokenize_single_sequence(tokenizer, antigen_seq, max_antigen_len)

    heavy_input_ids = heavy_enc["input_ids"].to(device)
    heavy_attention_mask = heavy_enc["attention_mask"].to(device)

    antigen_input_ids = antigen_enc["input_ids"].to(device)
    antigen_attention_mask = antigen_enc["attention_mask"].to(device)

    embedding_layer = model.get_input_embeddings()

    heavy_inputs_embeds = embedding_layer(heavy_input_ids).detach().clone()
    antigen_inputs_embeds = embedding_layer(antigen_input_ids).detach().clone()

    heavy_inputs_embeds.requires_grad_(True)
    antigen_inputs_embeds.requires_grad_(True)

    model.zero_grad(set_to_none=True)

    logits, heavy_to_antigen_attn, antigen_to_heavy_attn = model.forward_from_embeds(
        heavy_inputs_embeds=heavy_inputs_embeds,
        heavy_attention_mask=heavy_attention_mask,
        antigen_inputs_embeds=antigen_inputs_embeds,
        antigen_attention_mask=antigen_attention_mask,
    )

    pos_prob = torch.sigmoid(logits)[0]
    pos_prob.backward()

    heavy_grads = heavy_inputs_embeds.grad[0]      
    antigen_grads = antigen_inputs_embeds.grad[0]  
    heavy_embeds_0 = heavy_inputs_embeds.detach()[0]
    antigen_embeds_0 = antigen_inputs_embeds.detach()[0]

    if score_mode == "grad_norm":
        heavy_scores = torch.norm(heavy_grads, p=2, dim=-1)
        antigen_scores = torch.norm(antigen_grads, p=2, dim=-1)
    elif score_mode == "grad_x_input":
        heavy_scores = torch.sum(torch.abs(heavy_grads * heavy_embeds_0), dim=-1)
        antigen_scores = torch.sum(torch.abs(antigen_grads * antigen_embeds_0), dim=-1)
    else:
        raise ValueError("score_mode must be one of: 'grad_norm', 'grad_x_input'")

    heavy_scores = heavy_scores * heavy_attention_mask[0].float()
    antigen_scores = antigen_scores * antigen_attention_mask[0].float()

    if normalize:
        heavy_scores = normalize_scores(heavy_scores)
        antigen_scores = normalize_scores(antigen_scores)

    heavy_residue_importance = extract_residue_tokens_and_scores(
        tokenizer=tokenizer,
        input_ids=heavy_input_ids[0],
        attention_mask=heavy_attention_mask[0],
        scores=heavy_scores,
        original_seq=heavy_seq,
    )

    antigen_residue_importance = extract_residue_tokens_and_scores(
        tokenizer=tokenizer,
        input_ids=antigen_input_ids[0],
        attention_mask=antigen_attention_mask[0],
        scores=antigen_scores,
        original_seq=antigen_seq,
    )

    return {
        "logit": float(logits[0].item()),
        "positive_probability": float(pos_prob.item()),
        "score_mode": score_mode,
        "heavy_residue_importance": heavy_residue_importance,
        "antigen_residue_importance": antigen_residue_importance,
        "heavy_to_antigen_attn": heavy_to_antigen_attn.detach().cpu(),
        "antigen_to_heavy_attn": antigen_to_heavy_attn.detach().cpu(),
    }


def print_top_k_residues(residue_scores: List[Dict], title: str, top_k: int = 15):
    print(f"\n[{title}] top-{top_k}")
    sorted_items = sorted(
        residue_scores,
        key=lambda x: x["importance"],
        reverse=True
    )[:top_k]

    for item in sorted_items:
        print(
            f"pos={item['position_1based']:>4d} "
            f"res={item['residue']} "
            f"token={item['token']:<6s} "
            f"importance={item['importance']:.6f}"
        )


def save_residue_importance_csv(residue_scores: List[Dict], out_csv: str):
    df = pd.DataFrame(residue_scores)
    df.to_csv(out_csv, index=False)
    print(f"Saved attribution scores to: {out_csv}")


def train_main():
    set_seed(cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    dataset = PairDataset(
        csv_path=cfg.csv_path,
        heavy_col=cfg.heavy_col,
        antigen_col=cfg.antigen_col,
        label_col=cfg.label_col,
    )

    val_size = max(1, int(len(dataset) * cfg.val_ratio))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    collator = PairCollator(
        tokenizer=tokenizer,
        max_heavy_len=cfg.max_heavy_len,
        max_antigen_len=cfg.max_antigen_len,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collator,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collator,
    )

    model = ESM2BidirectionalCrossAttentionClassifier(
        model_name=cfg.model_name,
        hidden_dim=cfg.hidden_dim,
        num_heads=cfg.num_heads,
        dropout=cfg.dropout,
    ).to(cfg.device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    labels_all = [int(item["label"]) for item in dataset]
    pos = sum(labels_all)
    neg = len(labels_all) - pos

    if pos > 0:
        pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=cfg.device)
        print(f"Using pos_weight = {pos_weight.item():.4f}")
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    best_val_auc = -1.0

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, cfg.device
        )
        val_loss, val_metrics = evaluate(
            model, val_loader, criterion, cfg.device
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} "
            f"train_AUC={train_metrics['AUC']:.4f} "
            f"train_F1={train_metrics['F1']:.4f} "
            f"train_MCC={train_metrics['MCC']:.4f} "
            f"train_ACC={train_metrics['Accuracy']:.4f} | "
            f"val_loss={val_loss:.4f} "
            f"val_AUC={val_metrics['AUC']:.4f} "
            f"val_F1={val_metrics['F1']:.4f} "
            f"val_MCC={val_metrics['MCC']:.4f} "
            f"val_ACC={val_metrics['Accuracy']:.4f}"
        )

        if val_metrics["AUC"] > best_val_auc:
            best_val_auc = val_metrics["AUC"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": vars(cfg),
                    "best_val_auc": best_val_auc,
                },
                cfg.save_path,
            )
            print(f"Saved best model to {cfg.save_path}")


def load_trained_model(checkpoint_path: str, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device)

    model_cfg = ckpt.get("config", {})
    model_name = model_cfg.get("model_name", cfg.model_name)
    hidden_dim = model_cfg.get("hidden_dim", cfg.hidden_dim)
    num_heads = model_cfg.get("num_heads", cfg.num_heads)
    dropout = model_cfg.get("dropout", cfg.dropout)

    model = ESM2BidirectionalCrossAttentionClassifier(
        model_name=model_name,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout=dropout,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer, ckpt


def attribution_demo(
    checkpoint_path: str,
    heavy_seq: str,
    antigen_seq: str,
    score_mode: str = "grad_x_input",
):
    device = cfg.device
    model, tokenizer, ckpt = load_trained_model(checkpoint_path, device)

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Best val AUC: {ckpt.get('best_val_auc', 'N/A')}")

    pred = predict_single(
        model=model,
        tokenizer=tokenizer,
        heavy_seq=heavy_seq,
        antigen_seq=antigen_seq,
        device=device,
        max_heavy_len=cfg.max_heavy_len,
        max_antigen_len=cfg.max_antigen_len,
    )
    print(f"\nPrediction:")
    print(f"logit = {pred['logit']:.6f}")
    print(f"positive probability = {pred['prob']:.6f}")

    attr = attribute_positive_probability_to_input_embeddings(
        model=model,
        tokenizer=tokenizer,
        heavy_seq=heavy_seq,
        antigen_seq=antigen_seq,
        device=device,
        max_heavy_len=cfg.max_heavy_len,
        max_antigen_len=cfg.max_antigen_len,
        score_mode=score_mode,
        normalize=True,
    )

    print(f"\nAttribution mode: {attr['score_mode']}")
    print(f"logit = {attr['logit']:.6f}")
    print(f"positive_probability = {attr['positive_probability']:.6f}")

    print_top_k_residues(attr["heavy_residue_importance"], "Heavy chain", top_k=15)
    print_top_k_residues(attr["antigen_residue_importance"], "Antigen", top_k=15)

    save_residue_importance_csv(
        attr["heavy_residue_importance"],
        "heavy_residue_importance.csv"
    )
    save_residue_importance_csv(
        attr["antigen_residue_importance"],
        "antigen_residue_importance.csv"
    )

    return attr


if __name__ == "__main__":
 
    demo_heavy = "EVQLVESGGGLVQPGGSLRLSCAASGFTVSDNYMSWVRQAPGKGLQWVSVIYSGGNTYYADFVKGRFNITRDDSKNMLYLQMNSLRREDTAVYYCVRDRRIVGYYFGLDVWGQGTTVTVFS"
    demo_antigen = "RVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNF"
    
