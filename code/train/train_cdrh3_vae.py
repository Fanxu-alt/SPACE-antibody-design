from __future__ import annotations

import argparse
import hashlib
import platform
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split





def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
SPECIAL_TOKENS = ["<PAD>"]

ITOS = AMINO_ACIDS + SPECIAL_TOKENS
STOI = {token: index for index, token in enumerate(ITOS)}

PAD_IDX = STOI["<PAD>"]
VOCAB_SIZE = len(ITOS)
CANONICAL_AA = set(AMINO_ACIDS)




@dataclass
class Config:
    input_txt: str
    output_dir: str

    min_seq_len: int = 4
    max_seq_len: int = 30
    val_fraction: float = 0.10

    batch_size: int = 256
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    num_workers: int = 8

    embed_dim: int = 64
    hidden_dim: int = 128
    latent_dim: int = 64
    num_layers: int = 5
    kernel_size: int = 3
    dropout: float = 0.10

    beta_kl: float = 0.10
    kl_anneal_epochs: int = 10
    length_loss_weight: float = 0.20

    gradient_clip_norm: float = 5.0
    early_stopping_patience: int = 20

    seed: int = 42
    device: str = "cuda"




def clean_sequence(raw: str) -> str:
    """Normalize case/whitespace only; reject rather than mutate invalid residues."""
    return str(raw).strip().upper()


def load_unique_sequences(
    input_txt: str | Path,
    min_seq_len: int,
    max_seq_len: int,
) -> tuple[list[str], dict[str, int]]:
    input_path = Path(input_txt)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path.resolve()}")

    raw_count = 0
    empty_count = 0
    invalid_or_cleaned_count = 0
    length_filtered_count = 0
    accepted: list[str] = []

    with input_path.open("r", encoding="utf-8", errors="strict") as handle:
        for line in handle:
            raw_count += 1
            original = line.strip()
            if not original:
                empty_count += 1
                continue

            cleaned = clean_sequence(original)
            if any(aa not in CANONICAL_AA for aa in cleaned):
                invalid_or_cleaned_count += 1
                continue

            if not (min_seq_len <= len(cleaned) <= max_seq_len):
                length_filtered_count += 1
                continue

            accepted.append(cleaned)

    unique_sequences = sorted(set(accepted))

    if not unique_sequences:
        raise ValueError(
            "No valid CDRH3 sequences remained after cleaning and length filtering."
        )

    summary = {
        "raw_lines": raw_count,
        "empty_lines": empty_count,
        "lines_rejected_for_noncanonical_characters": invalid_or_cleaned_count,
        "length_filtered_lines": length_filtered_count,
        "accepted_before_deduplication": len(accepted),
        "unique_sequences": len(unique_sequences),
        "minimum_observed_length": min(map(len, unique_sequences)),
        "maximum_observed_length": max(map(len, unique_sequences)),
    }

    return unique_sequences, summary


class CDRH3TextDataset(Dataset):
    def __init__(self, sequences: list[str], max_seq_len: int):
        self.sequences = sequences
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.sequences)

    def encode(self, sequence: str) -> tuple[torch.Tensor, torch.Tensor]:
        true_length = min(len(sequence), self.max_seq_len)
        clipped = sequence[: self.max_seq_len]

        token_ids = [
            STOI[amino_acid]
            for amino_acid in clipped
        ]

        token_ids += [PAD_IDX] * (self.max_seq_len - len(token_ids))

        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(true_length, dtype=torch.long),
        )

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode(self.sequences[index])





class ResidualBlock1D(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dropout: float,
    ):
        super().__init__()

        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.norm1 = nn.BatchNorm1d(channels)

        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.norm2 = nn.BatchNorm1d(channels)

        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs

        hidden = self.conv1(inputs)
        hidden = self.norm1(hidden)
        hidden = F.relu(hidden, inplace=True)
        hidden = self.dropout(hidden)

        hidden = self.conv2(hidden)
        hidden = self.norm2(hidden)

        hidden = hidden + residual
        hidden = F.relu(hidden, inplace=True)

        return hidden


class CNNEncoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.embedding = nn.Embedding(
            VOCAB_SIZE,
            config.embed_dim,
            padding_idx=PAD_IDX,
        )

        self.input_projection = nn.Conv1d(
            config.embed_dim,
            config.hidden_dim,
            kernel_size=1,
        )

        self.blocks = nn.ModuleList(
            [
                ResidualBlock1D(
                    channels=config.hidden_dim,
                    kernel_size=config.kernel_size,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )

        flattened_dimension = config.hidden_dim * config.max_seq_len

        self.mu_head = nn.Linear(
            flattened_dimension,
            config.latent_dim,
        )
        self.logvar_head = nn.Linear(
            flattened_dimension,
            config.latent_dim,
        )

    def forward(
        self,
        tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embedding(tokens)
        hidden = embedded.transpose(1, 2)
        hidden = self.input_projection(hidden)

        for block in self.blocks:
            hidden = block(hidden)

        hidden = hidden.reshape(hidden.size(0), -1)

        mu = self.mu_head(hidden)
        logvar = self.logvar_head(hidden)

        return mu, logvar


class CNNDecoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.hidden_dim = config.hidden_dim
        self.max_seq_len = config.max_seq_len

        self.latent_projection = nn.Linear(
            config.latent_dim,
            config.hidden_dim * config.max_seq_len,
        )

        self.blocks = nn.ModuleList(
            [
                ResidualBlock1D(
                    channels=config.hidden_dim,
                    kernel_size=config.kernel_size,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )

        self.output_projection = nn.Conv1d(
            config.hidden_dim,
            VOCAB_SIZE,
            kernel_size=1,
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        hidden = self.latent_projection(latent)
        hidden = hidden.view(
            latent.size(0),
            self.hidden_dim,
            self.max_seq_len,
        )

        for block in self.blocks:
            hidden = block(hidden)

        logits = self.output_projection(hidden)
        return logits.transpose(1, 2)


class CDRH3CNNVAE(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.encoder = CNNEncoder(config)
        self.decoder = CNNDecoder(config)

        self.length_head = nn.Sequential(
            nn.Linear(config.latent_dim, config.latent_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.latent_dim, config.max_seq_len),
        )

    @staticmethod
    def reparameterize(
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        standard_deviation = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(standard_deviation)
        return mu + epsilon * standard_deviation

    def forward(
        self,
        tokens: torch.Tensor,
        sample_latent: bool = True,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        mu, logvar = self.encoder(tokens)
        latent = self.reparameterize(mu, logvar) if sample_latent else mu
        reconstruction_logits = self.decoder(latent)
        length_logits = self.length_head(latent)

        return (
            reconstruction_logits,
            mu,
            logvar,
            length_logits,
            latent,
        )




def compute_beta(
    epoch: int,
    maximum_beta: float,
    annealing_epochs: int,
) -> float:
    if annealing_epochs <= 0:
        return maximum_beta

    fraction = min(1.0, epoch / annealing_epochs)
    return maximum_beta * fraction


def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    true_lengths: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    length_logits: torch.Tensor,
    beta: float,
    length_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    reconstruction_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=PAD_IDX,
        reduction="mean",
    )

    kl_per_sequence = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp(),
        dim=1,
    )
    kl_loss = kl_per_sequence.mean()

 
    length_targets = true_lengths - 1
    length_loss = F.cross_entropy(
        length_logits,
        length_targets,
        reduction="mean",
    )

    total_loss = (
        reconstruction_loss
        + beta * kl_loss
        + length_loss_weight * length_loss
    )

    components = {
        "total": total_loss,
        "reconstruction": reconstruction_loss,
        "kl": kl_loss,
        "length": length_loss,
    }

    return total_loss, components


@torch.no_grad()
def compute_batch_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    length_logits: torch.Tensor,
    true_lengths: torch.Tensor,
) -> tuple[int, int, float, int]:
    predicted_tokens = logits.argmax(dim=-1)

    valid_mask = targets.ne(PAD_IDX)
    correct_tokens = (
        predicted_tokens.eq(targets)
        & valid_mask
    ).sum().item()
    total_tokens = valid_mask.sum().item()

    predicted_lengths = length_logits.argmax(dim=-1) + 1
    total_length_absolute_error = (
        predicted_lengths - true_lengths
    ).abs().sum().item()

    exact_length_matches = (
        predicted_lengths == true_lengths
    ).sum().item()

    return (
        int(correct_tokens),
        int(total_tokens),
        float(total_length_absolute_error),
        int(exact_length_matches),
    )





def run_epoch(
    model: CDRH3CNNVAE,
    loader: DataLoader,
    device: torch.device,
    beta: float,
    length_loss_weight: float,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip_norm: float,
) -> dict[str, float]:
    is_training = optimizer is not None
    model.train(is_training)

    loss_sums = {
        "total": 0.0,
        "reconstruction": 0.0,
        "kl": 0.0,
        "length": 0.0,
    }

    correct_tokens = 0
    total_tokens = 0
    total_length_absolute_error = 0.0
    exact_length_matches = 0
    total_sequences = 0
    number_of_batches = 0

    context = torch.enable_grad() if is_training else torch.no_grad()

    with context:
        for tokens, true_lengths in loader:
            tokens = tokens.to(
                device,
                non_blocking=True,
            )
            true_lengths = true_lengths.to(
                device,
                non_blocking=True,
            )

            if is_training:
                optimizer.zero_grad(set_to_none=True)

            (
                logits,
                mu,
                logvar,
                length_logits,
                _,
            ) = model(tokens, sample_latent=is_training)

            total_loss, components = compute_loss(
                logits=logits,
                targets=tokens,
                true_lengths=true_lengths,
                mu=mu,
                logvar=logvar,
                length_logits=length_logits,
                beta=beta,
                length_loss_weight=length_loss_weight,
            )

            if is_training:
                total_loss.backward()

                if gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=gradient_clip_norm,
                    )

                optimizer.step()

          
            for key in loss_sums:
                loss_sums[key] += components[key].item() * tokens.size(0)

            (
                batch_correct,
                batch_total,
                batch_length_absolute_error,
                batch_exact_length_matches,
            ) = compute_batch_metrics(
                logits=logits,
                targets=tokens,
                length_logits=length_logits,
                true_lengths=true_lengths,
            )

            correct_tokens += batch_correct
            total_tokens += batch_total
            total_length_absolute_error += batch_length_absolute_error
            exact_length_matches += batch_exact_length_matches
            total_sequences += tokens.size(0)
            number_of_batches += 1

    if number_of_batches == 0:
        raise RuntimeError("The DataLoader produced zero batches.")

    metrics = {
        f"{key}_loss": value / total_sequences
        for key, value in loss_sums.items()
    }
    metrics["token_accuracy"] = (
        correct_tokens / max(total_tokens, 1)
    )
    metrics["length_mae"] = (
        total_length_absolute_error / max(total_sequences, 1)
    )
    metrics["length_accuracy"] = (
        exact_length_matches / max(total_sequences, 1)
    )

    return metrics



def decode_tokens(
    token_ids: Iterable[int],
    output_length: int,
) -> str:
    characters: list[str] = []

    for token_id in list(token_ids)[:output_length]:
        token = ITOS[int(token_id)]
        if token == "<PAD>":
            continue
        characters.append(token)

    return "".join(characters)


@torch.no_grad()
def write_reconstruction_examples(
    model: CDRH3CNNVAE,
    dataset: Dataset,
    device: torch.device,
    output_path: Path,
    number_of_examples: int = 20,
) -> None:
    model.eval()

    lines = [
        "CDRH3 reconstruction examples",
        "=" * 80,
    ]

    example_count = min(number_of_examples, len(dataset))

    for index in range(example_count):
        tokens, true_length = dataset[index]
        batch_tokens = tokens.unsqueeze(0).to(device)

        logits, _, _, length_logits, _ = model(batch_tokens, sample_latent=False)

        predicted_tokens = (
            logits.argmax(dim=-1)
            .squeeze(0)
            .cpu()
            .tolist()
        )
        predicted_length = (
            int(length_logits.argmax(dim=-1).item())
            + 1
        )

        input_sequence = decode_tokens(
            tokens.tolist(),
            int(true_length.item()),
        )
        reconstructed_sequence = decode_tokens(
            predicted_tokens,
            predicted_length,
        )

        lines.extend(
            [
                f"Example {index + 1}",
                f"Input          ({int(true_length.item()):2d} aa): {input_sequence}",
                f"Reconstruction ({predicted_length:2d} aa): {reconstructed_sequence}",
                "-" * 80,
            ]
        )

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )





def save_checkpoint(
    output_path: Path,
    model: CDRH3CNNVAE,
    optimizer: torch.optim.Optimizer,
    config: Config,
    epoch: int,
    beta: float,
    train_metrics: dict[str, float],
    validation_metrics: dict[str, float],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "beta": beta,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
            "stoi": STOI,
            "itos": ITOS,
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": (lambda state: (state[0], state[1].tolist(), state[2], state[3], state[4]))(np.random.get_state()),
            "python_rng_state": random.getstate(),
            "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        output_path,
    )




def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pre-train a CNN-VAE on one-CDRH3-per-line OAS text data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input-txt",
        default="covid_human_heavy_cdr3_aa_unique_len4_30.txt",
    )
    parser.add_argument(
        "--output-dir",
        default="covid_cdrh3_vae_pretrain",
    )

    parser.add_argument("--min-seq-len", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=30)
    parser.add_argument("--val-fraction", type=float, default=0.10)

    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=8)

    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=5)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.10)

    parser.add_argument("--beta-kl", type=float, default=0.10)
    parser.add_argument("--kl-anneal-epochs", type=int, default=10)
    parser.add_argument("--length-loss-weight", type=float, default=0.20)

    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--early-stopping-patience", type=int, default=20)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
    )

    return parser




def main() -> None:
    arguments = build_parser().parse_args()

    config = Config(
        input_txt=arguments.input_txt,
        output_dir=arguments.output_dir,
        min_seq_len=arguments.min_seq_len,
        max_seq_len=arguments.max_seq_len,
        val_fraction=arguments.val_fraction,
        batch_size=arguments.batch_size,
        epochs=arguments.epochs,
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        num_workers=arguments.num_workers,
        embed_dim=arguments.embed_dim,
        hidden_dim=arguments.hidden_dim,
        latent_dim=arguments.latent_dim,
        num_layers=arguments.num_layers,
        kernel_size=arguments.kernel_size,
        dropout=arguments.dropout,
        beta_kl=arguments.beta_kl,
        kl_anneal_epochs=arguments.kl_anneal_epochs,
        length_loss_weight=arguments.length_loss_weight,
        gradient_clip_norm=arguments.gradient_clip_norm,
        early_stopping_patience=arguments.early_stopping_patience,
        seed=arguments.seed,
        device=arguments.device,
    )

    if config.min_seq_len < 1:
        raise ValueError("min_seq_len must be at least 1.")
    if config.max_seq_len < config.min_seq_len:
        raise ValueError("max_seq_len must be >= min_seq_len.")
    if not 0 < config.val_fraction < 1:
        raise ValueError("val_fraction must lie strictly between 0 and 1.")
    if config.batch_size < 1 or config.epochs < 1 or config.num_workers < 0:
        raise ValueError("batch_size and epochs must be positive; num_workers nonnegative.")
    if config.kernel_size < 1 or config.kernel_size % 2 == 0:
        raise ValueError("kernel_size should be odd to preserve sequence length.")

    set_seed(config.seed)

    output_directory = Path(config.output_dir)
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    with (
        output_directory / "config.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(
            asdict(config),
            handle,
            indent=2,
        )

    sequences, dataset_summary = load_unique_sequences(
        input_txt=config.input_txt,
        min_seq_len=config.min_seq_len,
        max_seq_len=config.max_seq_len,
    )

    # Record a cryptographic digest to identify the exact pretraining corpus.
    digest = hashlib.sha256()
    with Path(config.input_txt).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    dataset_summary["input_sha256"] = digest.hexdigest()
    dataset_summary["python_version"] = platform.python_version()
    dataset_summary["torch_version"] = torch.__version__
    dataset_summary["numpy_version"] = np.__version__
    dataset_summary["pandas_version"] = pd.__version__

    with (
        output_directory / "dataset_summary.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(
            dataset_summary,
            handle,
            indent=2,
        )

    dataset = CDRH3TextDataset(
        sequences=sequences,
        max_seq_len=config.max_seq_len,
    )

    validation_size = max(
        1,
        int(round(config.val_fraction * len(dataset))),
    )
    training_size = len(dataset) - validation_size

    if training_size < 1:
        raise ValueError("Training split is empty.")

    split_generator = torch.Generator().manual_seed(
        config.seed
    )

    training_dataset, validation_dataset = random_split(
        dataset,
        lengths=[
            training_size,
            validation_size,
        ],
        generator=split_generator,
    )


    np.savez_compressed(
        output_directory / "split_indices.npz",
        train_indices=np.asarray(training_dataset.indices, dtype=np.int64),
        val_indices=np.asarray(validation_dataset.indices, dtype=np.int64),
    )

    use_cuda = (
        config.device == "cuda"
        and torch.cuda.is_available()
    )

    if config.device == "cuda" and not use_cuda:
        raise RuntimeError(
            "CUDA was requested but no CUDA device is available."
        )

    device = torch.device(
        "cuda" if use_cuda else "cpu"
    )

    pin_memory = device.type == "cuda"
    persistent_workers = config.num_workers > 0

    training_loader = DataLoader(
        training_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    model = CDRH3CNNVAE(config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    print("=" * 88)
    print("COVID OAS CDRH3 CNN-VAE pre-training")
    print("=" * 88)
    print(f"Input file       : {Path(config.input_txt).resolve()}")
    print(f"Output directory : {output_directory.resolve()}")
    print(f"Unique sequences : {len(dataset):,}")
    print(f"Training set     : {training_size:,}")
    print(f"Validation set   : {validation_size:,}")
    print(f"Device           : {device}")
    if device.type == "cuda":
        print(f"GPU              : {torch.cuda.get_device_name(0)}")
    print(f"Vocabulary size  : {VOCAB_SIZE}")
    print(f"Maximum length   : {config.max_seq_len}")
    print(f"Latent dimension : {config.latent_dim}")
    print("=" * 88)

    best_validation_loss = math.inf
    epochs_without_improvement = 0
    history_rows: list[dict[str, float | int]] = []

    for epoch in range(1, config.epochs + 1):
        beta = compute_beta(
            epoch=epoch,
            maximum_beta=config.beta_kl,
            annealing_epochs=config.kl_anneal_epochs,
        )

        training_metrics = run_epoch(
            model=model,
            loader=training_loader,
            device=device,
            beta=beta,
            length_loss_weight=config.length_loss_weight,
            optimizer=optimizer,
            gradient_clip_norm=config.gradient_clip_norm,
        )

       
        validation_metrics = run_epoch(
            model=model,
            loader=validation_loader,
            device=device,
            beta=config.beta_kl,
            length_loss_weight=config.length_loss_weight,
            optimizer=None,
            gradient_clip_norm=0.0,
        )

        history_row: dict[str, float | int] = {
            "epoch": epoch,
            "beta": beta,
            "val_beta": config.beta_kl,
        }

        history_row.update(
            {
                f"train_{key}": value
                for key, value in training_metrics.items()
            }
        )
        history_row.update(
            {
                f"val_{key}": value
                for key, value in validation_metrics.items()
            }
        )
        history_rows.append(history_row)

        pd.DataFrame(history_rows).to_csv(
            output_directory / "training_history.csv",
            index=False,
        )

        save_checkpoint(
            output_path=output_directory / "last_model.pt",
            model=model,
            optimizer=optimizer,
            config=config,
            epoch=epoch,
            beta=beta,
            train_metrics=training_metrics,
            validation_metrics=validation_metrics,
        )

        print(
            f"Epoch {epoch:03d} | beta={beta:.4f} | "
            f"train total={training_metrics['total_loss']:.4f} "
            f"recon={training_metrics['reconstruction_loss']:.4f} "
            f"KL={training_metrics['kl_loss']:.4f} "
            f"len={training_metrics['length_loss']:.4f} "
            f"tok_acc={training_metrics['token_accuracy']:.4f} "
            f"len_MAE={training_metrics['length_mae']:.3f} | "
            f"val total={validation_metrics['total_loss']:.4f} "
            f"recon={validation_metrics['reconstruction_loss']:.4f} "
            f"KL={validation_metrics['kl_loss']:.4f} "
            f"len={validation_metrics['length_loss']:.4f} "
            f"tok_acc={validation_metrics['token_accuracy']:.4f} "
            f"len_MAE={validation_metrics['length_mae']:.3f}"
        )

        current_validation_loss = validation_metrics[
            "total_loss"
        ]

        if current_validation_loss < best_validation_loss:
            best_validation_loss = current_validation_loss
            epochs_without_improvement = 0

            save_checkpoint(
                output_path=output_directory / "best_model.pt",
                model=model,
                optimizer=optimizer,
                config=config,
                epoch=epoch,
                beta=beta,
                train_metrics=training_metrics,
                validation_metrics=validation_metrics,
            )

            print(
                f"Saved new best model: "
                f"val_total_loss={best_validation_loss:.6f}"
            )
        else:
            epochs_without_improvement += 1

        if (
            config.early_stopping_patience > 0
            and epochs_without_improvement
            >= config.early_stopping_patience
        ):
            print(
                "Early stopping triggered after "
                f"{config.early_stopping_patience} epochs "
                "without validation improvement."
            )
            break

    best_checkpoint = torch.load(
        output_directory / "best_model.pt",
        map_location=device,
    )
    model.load_state_dict(
        best_checkpoint["model_state_dict"]
    )

    write_reconstruction_examples(
        model=model,
        dataset=validation_dataset,
        device=device,
        output_path=(
            output_directory
            / "reconstruction_examples.txt"
        ),
        number_of_examples=20,
    )

    print("=" * 88)
    print("Training completed")
    print(f"Best validation loss: {best_validation_loss:.6f}")
    print(f"Best model: {(output_directory / 'best_model.pt').resolve()}")
    print("=" * 88)


if __name__ == "__main__":
    main()
