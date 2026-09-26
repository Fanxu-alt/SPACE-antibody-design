import os
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    matthews_corrcoef,
    accuracy_score,
)

from transformers import AutoTokenizer, AutoModel


@dataclass
class Config:
    split_dir: str = "splits"

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

    seeds: tuple = (42, 43, 44, 45, 46)

    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir: str = "cov_5run_results"


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

    def encode(self, input_ids, attention_mask):
        outputs = self.esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return outputs.last_hidden_state  # [B, L, D]

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

        logits = self.classifier(pair_feat).squeeze(-1)
        return logits, heavy_to_antigen_attn, antigen_to_heavy_attn



def compute_metrics(labels, probs, threshold=0.5):
    labels = np.array(labels).astype(int)
    probs = np.array(probs)
    preds = (probs >= threshold).astype(int)

    metrics = {}


    try:
        metrics["AUC"] = roc_auc_score(labels, probs)
    except ValueError:
        metrics["AUC"] = float("nan")

    metrics["F1"] = f1_score(labels, preds, zero_division=0)
    metrics["MCC"] = matthews_corrcoef(labels, preds)
    metrics["Accuracy"] = accuracy_score(labels, preds)

    return metrics



    model.train()

   
    model.esm.eval()

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



def main():

    os.makedirs(cfg.output_dir, exist_ok=True)

    checkpoint_dir = os.path.join(
        cfg.output_dir,
        "checkpoints"
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    all_results = []

    for run, seed in enumerate(cfg.seeds, start=1):

        print("\\n" + "=" * 80)
        print(f"RUN {run}/5 | seed={seed}")
        print("=" * 80)

        set_seed(seed)

        train_csv = os.path.join(
            cfg.split_dir,
            f"run{run}_seed{seed}_train.csv"
        )

        val_csv = os.path.join(
            cfg.split_dir,
            f"run{run}_seed{seed}_val.csv"
        )

        test_csv = os.path.join(
            cfg.split_dir,
            f"run{run}_seed{seed}_test.csv"
        )

        for path in [train_csv, val_csv, test_csv]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Missing split file: {path}"
                )

        print(f"Train: {train_csv}")
        print(f"Val:   {val_csv}")
        print(f"Test:  {test_csv}")


        train_set = PairDataset(
            train_csv,
            cfg.heavy_col,
            cfg.antigen_col,
            cfg.label_col,
        )

        val_set = PairDataset(
            val_csv,
            cfg.heavy_col,
            cfg.antigen_col,
            cfg.label_col,
        )

        test_set = PairDataset(
            test_csv,
            cfg.heavy_col,
            cfg.antigen_col,
            cfg.label_col,
        )

        print(
            f"Sizes | train={len(train_set)} "
            f"val={len(val_set)} "
            f"test={len(test_set)}"
        )

        assert len(train_set) == 21267
        assert len(val_set) == 2659
        assert len(test_set) == 2659

        collator = PairCollator(
            tokenizer=tokenizer,
            max_heavy_len=cfg.max_heavy_len,
            max_antigen_len=cfg.max_antigen_len,
        )

        g = torch.Generator()
        g.manual_seed(seed)

        train_loader = DataLoader(
            train_set,
            batch_size=cfg.batch_size,
            shuffle=True,
            generator=g,
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

        test_loader = DataLoader(
            test_set,
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
            [
                p for p in model.parameters()
                if p.requires_grad
            ],
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

       
        train_labels = [
            int(train_set[i]["label"])
            for i in range(len(train_set))
        ]

        pos_count = sum(train_labels)
        neg_count = len(train_labels) - pos_count

        if pos_count > 0:

            pos_weight = torch.tensor(
                [neg_count / pos_count],
                dtype=torch.float32,
                device=cfg.device,
            )

            criterion = nn.BCEWithLogitsLoss(
                pos_weight=pos_weight
            )

            print(
                f"Training labels | "
                f"positive={pos_count} "
                f"negative={neg_count} "
                f"pos_weight={pos_weight.item():.6f}"
            )

        else:
            criterion = nn.BCEWithLogitsLoss()

        checkpoint_path = os.path.join(
            checkpoint_dir,
            f"abagbinder_run{run}_seed{seed}.pt"
        )

        best_val_auc = -np.inf
        best_epoch = -1

        
        for epoch in range(1, cfg.epochs + 1):

            train_loss, train_metrics = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                cfg.device,
            )

            val_loss, val_metrics = evaluate(
                model,
                val_loader,
                criterion,
                cfg.device,
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


            if (
                not np.isnan(val_metrics["AUC"])
                and val_metrics["AUC"] > best_val_auc
            ):

                best_val_auc = val_metrics["AUC"]
                best_epoch = epoch

                torch.save(
                    {
                        "model_state_dict":
                            model.state_dict(),

                        "run": run,
                        "seed": seed,
                        "best_epoch": best_epoch,
                        "best_val_auc": best_val_auc,

                        "model_name": cfg.model_name,
                        "hidden_dim": cfg.hidden_dim,
                        "num_heads": cfg.num_heads,
                        "dropout": cfg.dropout,

                        "train_csv": train_csv,
                        "val_csv": val_csv,
                        "test_csv": test_csv,
                    },
                    checkpoint_path,
                )

                print(
                    f"Saved best checkpoint | "
                    f"epoch={best_epoch} "
                    f"val_AUC={best_val_auc:.6f}"
                )

        if best_epoch < 0:
            raise RuntimeError(
                f"No valid checkpoint for run {run}"
            )

       
        try:
            ckpt = torch.load(
                checkpoint_path,
                map_location=cfg.device,
                weights_only=False,
            )
        except TypeError:
            ckpt = torch.load(
                checkpoint_path,
                map_location=cfg.device,
            )

        model.load_state_dict(
            ckpt["model_state_dict"]
        )

        model.eval()

        
        test_loss, test_metrics = evaluate(
            model,
            test_loader,
            criterion,
            cfg.device,
        )

        print("\\nFINAL HELD-OUT TEST")

        print(
            f"Run={run} | "
            f"seed={seed} | "
            f"best_epoch={best_epoch} | "
            f"AUC={test_metrics['AUC']:.6f} | "
            f"F1={test_metrics['F1']:.6f} | "
            f"MCC={test_metrics['MCC']:.6f} | "
            f"Accuracy={test_metrics['Accuracy']:.6f}"
        )

        all_results.append(
            {
                "Run": run,
                "Seed": seed,
                "Best_epoch": best_epoch,
                "Best_val_AUC": best_val_auc,
                "Test_loss": test_loss,
                "AUC": test_metrics["AUC"],
                "F1": test_metrics["F1"],
                "MCC": test_metrics["MCC"],
                "Accuracy": test_metrics["Accuracy"],
            }
        )

        pd.DataFrame(all_results).to_csv(
            os.path.join(
                cfg.output_dir,
                "five_run_test_results.csv"
            ),
            index=False,
        )

        del model
        del optimizer

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    results_df = pd.DataFrame(all_results)

    print("\\n" + "=" * 80)
    print("FIVE-RUN HELD-OUT TEST RESULTS")
    print("=" * 80)

    print(
        results_df[
            [
                "Run",
                "Seed",
                "Best_epoch",
                "AUC",
                "F1",
                "MCC",
                "Accuracy",
            ]
        ].to_string(index=False)
    )

    summary_rows = []

    print("\\nMean +/- s.d. across five runs")
    print("-" * 80)

    for metric in [
        "AUC",
        "F1",
        "MCC",
        "Accuracy",
    ]:

        values = results_df[metric].astype(float)

        mean = values.mean()


        sd = values.std(ddof=1)

        summary_rows.append(
            {
                "Metric": metric,
                "Mean": mean,
                "SD": sd,
                "Mean_percent": mean * 100.0,
                "SD_percent": sd * 100.0,
            }
        )

        print(
            f"{metric:10s}: "
            f"{mean:.6f} +/- {sd:.6f} | "
            f"{mean*100:.2f} +/- {sd*100:.2f}%"
        )

    summary_df = pd.DataFrame(summary_rows)

    summary_df.to_csv(
        os.path.join(
            cfg.output_dir,
            "five_run_summary.csv"
        ),
        index=False,
    )

    print("\\nResults saved in:")
    print(cfg.output_dir)


if __name__ == "__main__":
    main()
