import os
import random
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
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

    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10

    num_runs: int = 5
    base_seed: int = 42

  
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

   
    output_dir: str = "results"


cfg = Config()


def set_seed(seed: int):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def masked_mean(x, mask):

    # x: [B, L, D]
    # mask: [B, L]

    mask = mask.unsqueeze(-1).float()

    x = x * mask

    return x.sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def safe_auc(labels, probs):

    try:
        return roc_auc_score(labels, probs)

    except ValueError:
        return float("nan")




class PairDataset(Dataset):

    def __init__(
        self,
        csv_path,
        heavy_col,
        antigen_col,
        label_col,
    ):

        df = pd.read_csv(csv_path)

        for col in [heavy_col, antigen_col, label_col]:

            if col not in df.columns:

                raise ValueError(
                    f"Column '{col}' not found in {csv_path}"
                )

        df = df[
            [heavy_col, antigen_col, label_col]
        ].dropna().copy()

        df[heavy_col] = (
            df[heavy_col]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        df[antigen_col] = (
            df[antigen_col]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        df[label_col] = pd.to_numeric(
            df[label_col],
            errors="coerce",
        )

        df = df.dropna().copy()

        df[label_col] = df[label_col].astype(int)

        df = df[
            (df[heavy_col].str.len() > 0)
            &
            (df[antigen_col].str.len() > 0)
        ].reset_index(drop=True)

        self.samples = list(
            zip(
                df[heavy_col].tolist(),
                df[antigen_col].tolist(),
                df[label_col].tolist(),
            )
        )

        if len(self.samples) == 0:

            raise ValueError(
                "No valid samples after filtering."
            )

        print(
            f"Loaded {len(self.samples)} samples"
        )

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

    def __init__(
        self,
        tokenizer,
        max_heavy_len,
        max_antigen_len,
    ):

        self.tokenizer = tokenizer

        self.max_heavy_len = max_heavy_len

        self.max_antigen_len = max_antigen_len

    @staticmethod
    def add_spaces(seq: str) -> str:

        return " ".join(list(seq))

    def __call__(self, batch):

        heavy_texts = [
            self.add_spaces(item["heavy"])
            for item in batch
        ]

        antigen_texts = [
            self.add_spaces(item["antigen"])
            for item in batch
        ]

        labels = torch.tensor(
            [item["label"] for item in batch],
            dtype=torch.float32,
        )

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

            "heavy_input_ids":
                heavy_inputs["input_ids"],

            "heavy_attention_mask":
                heavy_inputs["attention_mask"],

            "antigen_input_ids":
                antigen_inputs["input_ids"],

            "antigen_attention_mask":
                antigen_inputs["attention_mask"],

            "labels":
                labels,
        }



class CrossAttentionBlock(nn.Module):

    def __init__(
        self,
        dim,
        num_heads=8,
        dropout=0.1,
    ):

        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query,
        key_value,
        key_padding_mask=None,
    ):

        out, attn_weights = self.attn(

            query=query,

            key=key_value,

            value=key_value,

            key_padding_mask=key_padding_mask,

            need_weights=True,
        )

        out = self.norm(
            query + self.dropout(out)
        )

        return out, attn_weights



class ESM2BidirectionalCrossAttentionClassifier(nn.Module):

    def __init__(
        self,
        model_name,
        hidden_dim=256,
        num_heads=8,
        dropout=0.1,
    ):

        super().__init__()

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                model_name
            )
        )

        self.esm = (
            AutoModel.from_pretrained(
                model_name
            )
        )

 
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

        self.ab_proj = nn.Linear(
            esm_dim,
            hidden_dim,
        )

        self.ag_proj = nn.Linear(
            esm_dim,
            hidden_dim,
        )

        # [Ab, Ag, |Ab-Ag|, Ab*Ag]
        self.classifier = nn.Sequential(

            nn.Linear(
                hidden_dim * 4,
                hidden_dim,
            ),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(
                hidden_dim,
                1,
            ),
        )

    def get_input_embeddings(self):

        return self.esm.get_input_embeddings()

    def encode_from_ids(
        self,
        input_ids,
        attention_mask,
    ):

        outputs = self.esm(

            input_ids=input_ids,

            attention_mask=attention_mask,
        )

        return outputs.last_hidden_state

    def encode_from_embeds(
        self,
        inputs_embeds,
        attention_mask,
    ):

        outputs = self.esm(

            inputs_embeds=inputs_embeds,

            attention_mask=attention_mask,
        )

        return outputs.last_hidden_state

    def _forward_from_hidden(
        self,
        heavy_emb,
        heavy_attention_mask,
        antigen_emb,
        antigen_attention_mask,
    ):

        antigen_key_padding_mask = (
            antigen_attention_mask == 0
        )

        heavy_key_padding_mask = (
            heavy_attention_mask == 0
        )


        heavy_ctx, heavy_to_antigen_attn = (
            self.ab_to_ag(

                query=heavy_emb,

                key_value=antigen_emb,

                key_padding_mask=
                    antigen_key_padding_mask,
            )
        )


        antigen_ctx, antigen_to_heavy_attn = (
            self.ag_to_ab(

                query=antigen_emb,

                key_value=heavy_emb,

                key_padding_mask=
                    heavy_key_padding_mask,
            )
        )

   
        heavy_vec = masked_mean(

            self.ab_proj(heavy_ctx),

            heavy_attention_mask,
        )

        antigen_vec = masked_mean(

            self.ag_proj(antigen_ctx),

            antigen_attention_mask,
        )

       
        pair_feat = torch.cat(
            [
                heavy_vec,

                antigen_vec,

                torch.abs(
                    heavy_vec - antigen_vec
                ),

                heavy_vec * antigen_vec,
            ],
            dim=-1,
        )

        logits = (
            self.classifier(pair_feat)
            .squeeze(-1)
        )

        return (
            logits,
            heavy_to_antigen_attn,
            antigen_to_heavy_attn,
        )

    def forward(
        self,
        heavy_input_ids,
        heavy_attention_mask,
        antigen_input_ids,
        antigen_attention_mask,
    ):

     
        with torch.no_grad():

            heavy_emb = self.encode_from_ids(
                heavy_input_ids,
                heavy_attention_mask,
            )

            antigen_emb = self.encode_from_ids(
                antigen_input_ids,
                antigen_attention_mask,
            )

        return self._forward_from_hidden(

            heavy_emb=heavy_emb,

            heavy_attention_mask=
                heavy_attention_mask,

            antigen_emb=antigen_emb,

            antigen_attention_mask=
                antigen_attention_mask,
        )

    def forward_from_embeds(
        self,
        heavy_inputs_embeds,
        heavy_attention_mask,
        antigen_inputs_embeds,
        antigen_attention_mask,
    ):

        heavy_emb = self.encode_from_embeds(
            heavy_inputs_embeds,
            heavy_attention_mask,
        )

        antigen_emb = self.encode_from_embeds(
            antigen_inputs_embeds,
            antigen_attention_mask,
        )

        return self._forward_from_hidden(

            heavy_emb=heavy_emb,

            heavy_attention_mask=
                heavy_attention_mask,

            antigen_emb=antigen_emb,

            antigen_attention_mask=
                antigen_attention_mask,
        )



def compute_metrics(
    labels,
    probs,
    threshold=0.5,
):

    labels = np.asarray(
        labels
    ).astype(int)

    probs = np.asarray(
        probs
    )

    preds = (
        probs >= threshold
    ).astype(int)

    return {

        "AUC":
            safe_auc(labels, probs),

        "F1":
            f1_score(
                labels,
                preds,
                zero_division=0,
            ),

        "MCC":
            matthews_corrcoef(
                labels,
                preds,
            ),

        "Accuracy":
            accuracy_score(
                labels,
                preds,
            ),
    }




def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
):

    model.train()

    total_loss = 0.0

    all_probs = []

    all_labels = []

    for batch in loader:

        heavy_input_ids = (
            batch["heavy_input_ids"]
            .to(device)
        )

        heavy_attention_mask = (
            batch["heavy_attention_mask"]
            .to(device)
        )

        antigen_input_ids = (
            batch["antigen_input_ids"]
            .to(device)
        )

        antigen_attention_mask = (
            batch["antigen_attention_mask"]
            .to(device)
        )

        labels = (
            batch["labels"]
            .to(device)
        )

        optimizer.zero_grad()

        logits, _, _ = model(

            heavy_input_ids=
                heavy_input_ids,

            heavy_attention_mask=
                heavy_attention_mask,

            antigen_input_ids=
                antigen_input_ids,

            antigen_attention_mask=
                antigen_attention_mask,
        )

        loss = criterion(
            logits,
            labels,
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        probs = torch.sigmoid(logits)

        all_probs.extend(
            probs.detach()
            .cpu()
            .numpy()
            .tolist()
        )

        all_labels.extend(
            labels.detach()
            .cpu()
            .numpy()
            .tolist()
        )

    metrics = compute_metrics(
        all_labels,
        all_probs,
    )

    return (
        total_loss / len(loader),
        metrics,
    )



@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    total_loss = 0.0

    all_probs = []

    all_labels = []

    for batch in loader:

        heavy_input_ids = (
            batch["heavy_input_ids"]
            .to(device)
        )

        heavy_attention_mask = (
            batch["heavy_attention_mask"]
            .to(device)
        )

        antigen_input_ids = (
            batch["antigen_input_ids"]
            .to(device)
        )

        antigen_attention_mask = (
            batch["antigen_attention_mask"]
            .to(device)
        )

        labels = (
            batch["labels"]
            .to(device)
        )

        logits, _, _ = model(

            heavy_input_ids=
                heavy_input_ids,

            heavy_attention_mask=
                heavy_attention_mask,

            antigen_input_ids=
                antigen_input_ids,

            antigen_attention_mask=
                antigen_attention_mask,
        )

        loss = criterion(
            logits,
            labels,
        )

        total_loss += loss.item()

        probs = torch.sigmoid(logits)

        all_probs.extend(
            probs.cpu()
            .numpy()
            .tolist()
        )

        all_labels.extend(
            labels.cpu()
            .numpy()
            .tolist()
        )

    metrics = compute_metrics(
        all_labels,
        all_probs,
    )

    return (
        total_loss / len(loader),
        metrics,
    )



def make_split_indices(
    dataset,
    seed,
):

    indices = np.arange(
        len(dataset)
    )

    labels = np.array(
        [
            int(dataset[i]["label"])
            for i in indices
        ]
    )



    train_idx, temp_idx = train_test_split(

        indices,

        test_size=0.20,

        random_state=seed,

        stratify=labels,
    )

    temp_labels = labels[temp_idx]

  

    val_idx, test_idx = train_test_split(

        temp_idx,

        test_size=0.50,

        random_state=seed,

        stratify=temp_labels,
    )

    return (
        np.asarray(train_idx),
        np.asarray(val_idx),
        np.asarray(test_idx),
    )



def print_split_statistics(
    dataset,
    indices,
    name,
):

    labels = np.array(
        [
            int(dataset[int(i)]["label"])
            for i in indices
        ]
    )

    n = len(labels)

    pos = int(labels.sum())

    neg = n - pos

    ratio = pos / n if n > 0 else 0

    print(
        f"{name:<12} "
        f"N={n:<7} "
        f"Positive={pos:<7} "
        f"Negative={neg:<7} "
        f"Positive ratio={ratio:.4f}"
    )




def build_training_criterion(
    dataset,
    train_idx,
    device,
):

    train_labels = np.array(
        [
            int(
                dataset[int(i)]["label"]
            )
            for i in train_idx
        ]
    )

    pos = int(
        train_labels.sum()
    )

    neg = (
        len(train_labels) - pos
    )

    if pos > 0:

        pos_weight_value = (
            neg / pos
        )

        pos_weight = torch.tensor(
            [pos_weight_value],
            dtype=torch.float32,
            device=device,
        )

        criterion = (
            nn.BCEWithLogitsLoss(
                pos_weight=pos_weight
            )
        )

    else:

        pos_weight_value = 1.0

        criterion = (
            nn.BCEWithLogitsLoss()
        )

    return (
        criterion,
        pos_weight_value,
    )



def train_main():

    # --------------------------------------------------------
    # Output directories
    # --------------------------------------------------------

    split_dir = os.path.join(
        cfg.output_dir,
        "splits",
    )

    checkpoint_dir = os.path.join(
        cfg.output_dir,
        "checkpoints",
    )

    os.makedirs(
        split_dir,
        exist_ok=True,
    )

    os.makedirs(
        checkpoint_dir,
        exist_ok=True,
    )


    dataset = PairDataset(

        csv_path=cfg.csv_path,

        heavy_col=cfg.heavy_col,

        antigen_col=cfg.antigen_col,

        label_col=cfg.label_col,
    )

    print(
        f"\nDevice: {cfg.device}"
    )

    print(
        f"Number of runs: {cfg.num_runs}"
    )

    print(
        "Split: 80% train / "
        "10% validation / "
        "10% held-out test"
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            cfg.model_name
        )
    )

    collator = PairCollator(

        tokenizer=tokenizer,

        max_heavy_len=
            cfg.max_heavy_len,

        max_antigen_len=
            cfg.max_antigen_len,
    )

    all_results = []



    for run_idx in range(
        cfg.num_runs
    ):

        run_number = (
            run_idx + 1
        )

        seed = (
            cfg.base_seed
            + run_idx
        )

        print(
            "\n"
            + "=" * 80
        )

        print(
            f"RUN {run_number}/"
            f"{cfg.num_runs}"
        )

        print(
            f"Random seed: {seed}"
        )

        print(
            "=" * 80
        )


        set_seed(seed)

 

        (
            train_idx,
            val_idx,
            test_idx,
        ) = make_split_indices(
            dataset,
            seed,
        )



        assert (
            len(
                set(train_idx)
                &
                set(val_idx)
            )
            == 0
        )

        assert (
            len(
                set(train_idx)
                &
                set(test_idx)
            )
            == 0
        )

        assert (
            len(
                set(val_idx)
                &
                set(test_idx)
            )
            == 0
        )

        assert (
            len(train_idx)
            + len(val_idx)
            + len(test_idx)
            ==
            len(dataset)
        )

  

        split_path = os.path.join(
            split_dir,
            f"split_run{run_number}.npz",
        )

        np.savez(

            split_path,

            train_idx=train_idx,

            val_idx=val_idx,

            test_idx=test_idx,

            seed=np.array([seed]),
        )

        print(
            f"\nSaved split -> "
            f"{split_path}"
        )


        print(
            "\nSplit statistics:"
        )

        print_split_statistics(
            dataset,
            train_idx,
            "Train",
        )

        print_split_statistics(
            dataset,
            val_idx,
            "Validation",
        )

        print_split_statistics(
            dataset,
            test_idx,
            "Test",
        )

  

        train_set = Subset(
            dataset,
            train_idx.tolist(),
        )

        val_set = Subset(
            dataset,
            val_idx.tolist(),
        )

        test_set = Subset(
            dataset,
            test_idx.tolist(),
        )

       


        train_generator = (
            torch.Generator()
            .manual_seed(seed)
        )

        train_loader = DataLoader(

            train_set,

            batch_size=
                cfg.batch_size,

            shuffle=True,

            generator=
                train_generator,

            num_workers=
                cfg.num_workers,

            collate_fn=
                collator,
        )

        val_loader = DataLoader(

            val_set,

            batch_size=
                cfg.batch_size,

            shuffle=False,

            num_workers=
                cfg.num_workers,

            collate_fn=
                collator,
        )

        test_loader = DataLoader(

            test_set,

            batch_size=
                cfg.batch_size,

            shuffle=False,

            num_workers=
                cfg.num_workers,

            collate_fn=
                collator,
        )

     

        model = (
            ESM2BidirectionalCrossAttentionClassifier(

                model_name=
                    cfg.model_name,

                hidden_dim=
                    cfg.hidden_dim,

                num_heads=
                    cfg.num_heads,

                dropout=
                    cfg.dropout,
            )
            .to(cfg.device)
        )

        optimizer = torch.optim.AdamW(

            [
                p
                for p
                in model.parameters()
                if p.requires_grad
            ],

            lr=cfg.lr,

            weight_decay=
                cfg.weight_decay,
        )

        

        (
            criterion,
            pos_weight,
        ) = build_training_criterion(

            dataset,

            train_idx,

            cfg.device,
        )

        print(
            f"\nTraining pos_weight: "
            f"{pos_weight:.6f}"
        )

    

        best_val_auc = -np.inf

        best_epoch = -1

        checkpoint_path = os.path.join(

            checkpoint_dir,

            f"best_run{run_number}.pt",
        )



        for epoch in range(
            1,
            cfg.epochs + 1,
        ):

            (
                train_loss,
                train_metrics,
            ) = train_one_epoch(

                model,

                train_loader,

                optimizer,

                criterion,

                cfg.device,
            )

            (
                val_loss,
                val_metrics,
            ) = evaluate(

                model,

                val_loader,

                criterion,

                cfg.device,
            )

            print(

                f"Epoch "
                f"{epoch:02d}/{cfg.epochs} | "

                f"Train loss="
                f"{train_loss:.4f} | "

                f"Train AUC="
                f"{train_metrics['AUC']:.4f} | "

                f"Val loss="
                f"{val_loss:.4f} | "

                f"Val AUC="
                f"{val_metrics['AUC']:.4f} | "

                f"Val F1="
                f"{val_metrics['F1']:.4f} | "

                f"Val MCC="
                f"{val_metrics['MCC']:.4f} | "

                f"Val ACC="
                f"{val_metrics['Accuracy']:.4f}"
            )

            current_val_auc = (
                val_metrics["AUC"]
            )

            if (
                not np.isnan(
                    current_val_auc
                )
                and
                current_val_auc
                >
                best_val_auc
            ):

                best_val_auc = (
                    current_val_auc
                )

                best_epoch = epoch

                torch.save(

                    {
                        "run":
                            run_number,

                        "seed":
                            seed,

                        "epoch":
                            epoch,

                        "model_state_dict":
                            model.state_dict(),

                        "optimizer_state_dict":
                            optimizer.state_dict(),

                        "config":
                            vars(cfg),

                        "best_val_auc":
                            best_val_auc,

                        "train_idx":
                            train_idx,

                        "val_idx":
                            val_idx,

                        "test_idx":
                            test_idx,
                    },

                    checkpoint_path,
                )

                print(
                    f"  -> Best checkpoint "
                    f"saved "
                    f"(Val AUC="
                    f"{best_val_auc:.4f})"
                )



        print(
            "\nLoading best validation checkpoint..."
        )

        checkpoint = torch.load(

            checkpoint_path,

            map_location=
                cfg.device,
        )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        model.eval()

        (
            test_loss,
            test_metrics,
        ) = evaluate(

            model,

            test_loader,

            criterion,

            cfg.device,
        )

        print(
            "\n"
            + "-" * 80
        )

        print(
            f"RUN {run_number} "
            f"HELD-OUT TEST RESULTS"
        )

        print(
            "-" * 80
        )

        print(
            f"Seed          : {seed}"
        )

        print(
            f"Best epoch    : "
            f"{best_epoch}"
        )

        print(
            f"Best Val AUC  : "
            f"{best_val_auc:.6f}"
        )

        print(
            f"Test Loss     : "
            f"{test_loss:.6f}"
        )

        print(
            f"Test AUC      : "
            f"{test_metrics['AUC']:.6f}"
        )

        print(
            f"Test F1       : "
            f"{test_metrics['F1']:.6f}"
        )

        print(
            f"Test MCC      : "
            f"{test_metrics['MCC']:.6f}"
        )

        print(
            f"Test Accuracy : "
            f"{test_metrics['Accuracy']:.6f}"
        )


        all_results.append(

            {
                "Run":
                    run_number,

                "Seed":
                    seed,

                "Train_N":
                    len(train_idx),

                "Validation_N":
                    len(val_idx),

                "Test_N":
                    len(test_idx),

                "Best_Epoch":
                    best_epoch,

                "Best_Validation_AUC":
                    best_val_auc,

                "Test_Loss":
                    test_loss,

                "Test_AUC":
                    test_metrics["AUC"],

                "Test_F1":
                    test_metrics["F1"],

                "Test_MCC":
                    test_metrics["MCC"],

                "Test_Accuracy":
                    test_metrics["Accuracy"],
            }
        )



        del model

        if torch.cuda.is_available():

            torch.cuda.empty_cache()



    results_df = pd.DataFrame(
        all_results
    )

    results_path = os.path.join(
        cfg.output_dir,
        "five_run_results.csv",
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "FIVE-RUN HELD-OUT TEST SUMMARY"
    )

    print(
        "=" * 80
    )

    print(
        results_df.to_string(
            index=False
        )
    )



    metric_columns = [

        "Test_AUC",

        "Test_F1",

        "Test_MCC",

        "Test_Accuracy",
    ]

    summary_rows = []

    print(
        "\nMean ± SD across "
        "five independent runs:"
    )

    for metric in metric_columns:

        values = (
            results_df[metric]
            .astype(float)
            .to_numpy()
        )

        mean_value = (
            np.nanmean(values)
        )

        sd_value = (
            np.nanstd(
                values,
                ddof=1,
            )
        )

        summary_rows.append(

            {
                "Metric":
                    metric,

                "Mean":
                    mean_value,

                "SD":
                    sd_value,
            }
        )

        print(
            f"{metric:<15}: "
            f"{mean_value:.4f} "
            f"± "
            f"{sd_value:.4f}"
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_path = os.path.join(
        cfg.output_dir,
        "five_run_summary.csv",
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    print(
        f"\nPer-run results saved to:\n"
        f"{results_path}"
    )

    print(
        f"\nSummary saved to:\n"
        f"{summary_path}"
    )

    print(
        f"\nSplits saved to:\n"
        f"{split_dir}"
    )

    print(
        f"\nCheckpoints saved to:\n"
        f"{checkpoint_dir}"
    )

    return (
        results_df,
        summary_df,
    )



if __name__ == "__main__":

    train_main()
