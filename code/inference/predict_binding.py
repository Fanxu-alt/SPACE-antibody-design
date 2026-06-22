import argparse
from typing import List, Dict, Optional

import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


def add_spaces(seq: str) -> str:
    return " ".join(list(seq.strip().upper()))


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.unsqueeze(-1).float()
    x = x * mask
    return x.sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
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


class ESM2BidirectionalCrossAttentionClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

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

    def encode_from_ids(self, input_ids, attention_mask):
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
            heavy_emb = self.encode_from_ids(
                heavy_input_ids,
                heavy_attention_mask,
            )
            antigen_emb = self.encode_from_ids(
                antigen_input_ids,
                antigen_attention_mask,
            )

        antigen_key_padding_mask = antigen_attention_mask == 0
        heavy_key_padding_mask = heavy_attention_mask == 0

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
                torch.abs(heavy_vec - antigen_vec),
                heavy_vec * antigen_vec,
            ],
            dim=-1,
        )

        logits = self.classifier(pair_feat).squeeze(-1)

        return logits, heavy_to_antigen_attn, antigen_to_heavy_attn


def load_model(
    checkpoint_path: str,
    device: str,
    fallback_model_name: str = "facebook/esm2_t6_8M_UR50D",
):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_config = checkpoint.get("config", {})

    model_name = model_config.get("model_name", fallback_model_name)
    hidden_dim = int(model_config.get("hidden_dim", 256))
    num_heads = int(model_config.get("num_heads", 8))
    dropout = float(model_config.get("dropout", 0.1))

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = ESM2BidirectionalCrossAttentionClassifier(
        model_name=model_name,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout=dropout,
    ).to(device)

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return model, tokenizer, model_config


def tokenize_sequences(
    tokenizer,
    sequences: List[str],
    max_length: int,
    device: str,
):
    texts = [add_spaces(seq) for seq in sequences]

    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    return {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
    }


@torch.no_grad()
def predict_batch(
    model,
    tokenizer,
    heavy_sequences: List[str],
    antigen_sequences: List[str],
    device: str,
    max_heavy_len: int = 256,
    max_antigen_len: int = 512,
):
    heavy_inputs = tokenize_sequences(
        tokenizer=tokenizer,
        sequences=heavy_sequences,
        max_length=max_heavy_len,
        device=device,
    )

    antigen_inputs = tokenize_sequences(
        tokenizer=tokenizer,
        sequences=antigen_sequences,
        max_length=max_antigen_len,
        device=device,
    )

    logits, _, _ = model(
        heavy_input_ids=heavy_inputs["input_ids"],
        heavy_attention_mask=heavy_inputs["attention_mask"],
        antigen_input_ids=antigen_inputs["input_ids"],
        antigen_attention_mask=antigen_inputs["attention_mask"],
    )

    probs = torch.sigmoid(logits)

    return logits.cpu().numpy(), probs.cpu().numpy()


def predict_from_csv(
    model,
    tokenizer,
    input_csv: str,
    output_csv: str,
    heavy_col: str,
    antigen_col: str,
    device: str,
    batch_size: int,
    max_heavy_len: int,
    max_antigen_len: int,
):
    df = pd.read_csv(input_csv)

    if heavy_col not in df.columns:
        raise ValueError(f"Heavy-chain column '{heavy_col}' not found in {input_csv}")

    if antigen_col not in df.columns:
        raise ValueError(f"Antigen column '{antigen_col}' not found in {input_csv}")

    df = df.copy()
    df[heavy_col] = df[heavy_col].astype(str).str.strip().str.upper()
    df[antigen_col] = df[antigen_col].astype(str).str.strip().str.upper()

    all_logits = []
    all_probs = []

    for start in range(0, len(df), batch_size):
        end = start + batch_size

        heavy_batch = df[heavy_col].iloc[start:end].tolist()
        antigen_batch = df[antigen_col].iloc[start:end].tolist()

        logits, probs = predict_batch(
            model=model,
            tokenizer=tokenizer,
            heavy_sequences=heavy_batch,
            antigen_sequences=antigen_batch,
            device=device,
            max_heavy_len=max_heavy_len,
            max_antigen_len=max_antigen_len,
        )

        all_logits.extend(logits.tolist())
        all_probs.extend(probs.tolist())

    df["binding_logit"] = all_logits
    df["binding_probability"] = all_probs
    df["predicted_label"] = (df["binding_probability"] >= 0.5).astype(int)

    df.to_csv(output_csv, index=False)
    print(f"Saved predictions to: {output_csv}")


def predict_single_pair(
    model,
    tokenizer,
    heavy_seq: str,
    antigen_seq: str,
    device: str,
    max_heavy_len: int,
    max_antigen_len: int,
):
    logits, probs = predict_batch(
        model=model,
        tokenizer=tokenizer,
        heavy_sequences=[heavy_seq],
        antigen_sequences=[antigen_seq],
        device=device,
        max_heavy_len=max_heavy_len,
        max_antigen_len=max_antigen_len,
    )

    print("Prediction result")
    print("-----------------")
    print(f"Binding logit: {float(logits[0]):.6f}")
    print(f"Binding probability: {float(probs[0]):.6f}")
    print(f"Predicted label: {int(probs[0] >= 0.5)}")


def main():
    parser = argparse.ArgumentParser(
        description="Predict antibody-antigen binding probability using AbAgBinder."
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained AbAgBinder checkpoint.",
    )

    parser.add_argument(
        "--input_csv",
        type=str,
        default=None,
        help="Input CSV containing antibody heavy-chain and antigen sequences.",
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        default="binding_predictions.csv",
        help="Output CSV path.",
    )

    parser.add_argument(
        "--heavy_col",
        type=str,
        default="Heavy",
        help="Column name for antibody heavy-chain sequence.",
    )

    parser.add_argument(
        "--antigen_col",
        type=str,
        default="antigen",
        help="Column name for antigen sequence.",
    )

    parser.add_argument(
        "--heavy_seq",
        type=str,
        default=None,
        help="Single antibody heavy-chain sequence.",
    )

    parser.add_argument(
        "--antigen_seq",
        type=str,
        default=None,
        help="Single antigen sequence.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for CSV prediction.",
    )

    parser.add_argument(
        "--max_heavy_len",
        type=int,
        default=256,
        help="Maximum token length for antibody heavy chain.",
    )

    parser.add_argument(
        "--max_antigen_len",
        type=int,
        default=512,
        help="Maximum token length for antigen.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device: cuda, cpu, or mps.",
    )

    args = parser.parse_args()

    model, tokenizer, model_config = load_model(
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    if args.input_csv is not None:
        predict_from_csv(
            model=model,
            tokenizer=tokenizer,
            input_csv=args.input_csv,
            output_csv=args.output_csv,
            heavy_col=args.heavy_col,
            antigen_col=args.antigen_col,
            device=args.device,
            batch_size=args.batch_size,
            max_heavy_len=args.max_heavy_len,
            max_antigen_len=args.max_antigen_len,
        )

    elif args.heavy_seq is not None and args.antigen_seq is not None:
        predict_single_pair(
            model=model,
            tokenizer=tokenizer,
            heavy_seq=args.heavy_seq,
            antigen_seq=args.antigen_seq,
            device=args.device,
            max_heavy_len=args.max_heavy_len,
            max_antigen_len=args.max_antigen_len,
        )

    else:
        raise ValueError(
            "Please provide either --input_csv or both --heavy_seq and --antigen_seq."
        )


if __name__ == "__main__":
    main()
