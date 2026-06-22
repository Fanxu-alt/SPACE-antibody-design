import torch
import torch.nn as nn
import torch.nn.functional as F


CKPT_PATH = "conditional_cvae_finetune.pt"

TARGET_ANTIGEN = (
    "RVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNF"
)

NUM_SAMPLES = 1000
MIN_LEN = 8
TEMPERATURE = 1.0
SAMPLE_MODE = "sample"   # "sample" or "argmax"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
SPECIAL_TOKENS = ["X", "<PAD>"]

itos = AMINO_ACIDS + SPECIAL_TOKENS
stoi = {ch: i for i, ch in enumerate(itos)}

PAD_IDX = stoi["<PAD>"]
UNK_IDX = stoi["X"]
VOCAB_SIZE = len(itos)


class ResBlock1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        out = F.relu(out, inplace=True)
        return out


class CNNEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        latent_dim: int,
        max_seq_len: int,
        num_layers: int = 5,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.input_proj = nn.Conv1d(embed_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList(
            [ResBlock1D(hidden_dim, kernel_size, dropout) for _ in range(num_layers)]
        )

        flat_dim = hidden_dim * max_seq_len
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

    def encode_feature(self, x):
        emb = self.embedding(x)
        emb = emb.transpose(1, 2)
        h = self.input_proj(emb)
        for block in self.blocks:
            h = block(h)
        h = h.reshape(h.size(0), -1)
        return h

    def forward(self, x):
        h = self.encode_feature(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class CNNDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        vocab_size: int,
        max_seq_len: int,
        num_layers: int = 5,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim

        self.fc = nn.Linear(latent_dim, hidden_dim * max_seq_len)
        self.blocks = nn.ModuleList(
            [ResBlock1D(hidden_dim, kernel_size, dropout) for _ in range(num_layers)]
        )
        self.output_proj = nn.Conv1d(hidden_dim, vocab_size, kernel_size=1)

    def forward(self, z):
        h = self.fc(z)
        h = h.view(z.size(0), self.hidden_dim, self.max_seq_len)
        for block in self.blocks:
            h = block(h)
        logits = self.output_proj(h)
        logits = logits.transpose(1, 2)
        return logits


class AntigenEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.input_proj = nn.Conv1d(embed_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList(
            [ResBlock1D(hidden_dim, kernel_size, dropout) for _ in range(num_layers)]
        )

    def forward(self, a, a_mask):
        emb = self.embedding(a)
        emb = emb.transpose(1, 2)
        h = self.input_proj(emb)
        for block in self.blocks:
            h = block(h)

        mask = a_mask.unsqueeze(1).float()
        h = h * mask
        pooled = h.sum(dim=2) / mask.sum(dim=2).clamp_min(1.0)
        return pooled


class ConditionalCNNVAE(nn.Module):
    def __init__(self, config: dict):
        super().__init__()

        self.encoder = CNNEncoder(
            vocab_size=VOCAB_SIZE,
            embed_dim=config["embed_dim"],
            hidden_dim=config["hidden_dim"],
            latent_dim=config["latent_dim"],
            max_seq_len=config["max_cdr3_len"],
            num_layers=config["num_layers"],
            kernel_size=config["kernel_size"],
            dropout=config["dropout"],
        )

        self.decoder = CNNDecoder(
            latent_dim=config["latent_dim"],
            hidden_dim=config["hidden_dim"],
            vocab_size=VOCAB_SIZE,
            max_seq_len=config["max_cdr3_len"],
            num_layers=config["num_layers"],
            kernel_size=config["kernel_size"],
            dropout=config["dropout"],
        )

        self.antigen_encoder = AntigenEncoder(
            vocab_size=VOCAB_SIZE,
            embed_dim=config["embed_dim"],
            hidden_dim=config["antigen_hidden_dim"],
            num_layers=config["antigen_num_layers"],
            kernel_size=config["kernel_size"],
            dropout=config["dropout"],
        )

        seq_feat_dim = config["hidden_dim"] * config["max_cdr3_len"]
        ant_feat_dim = config["antigen_hidden_dim"]

        self.posterior_mu = nn.Sequential(
            nn.Linear(seq_feat_dim + ant_feat_dim, config["fusion_dim"]),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["fusion_dim"], config["latent_dim"]),
        )
        self.posterior_logvar = nn.Sequential(
            nn.Linear(seq_feat_dim + ant_feat_dim, config["fusion_dim"]),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["fusion_dim"], config["latent_dim"]),
        )

        self.prior_mu = nn.Sequential(
            nn.Linear(ant_feat_dim, config["fusion_dim"]),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["fusion_dim"], config["latent_dim"]),
        )
        self.prior_logvar = nn.Sequential(
            nn.Linear(ant_feat_dim, config["fusion_dim"]),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["fusion_dim"], config["latent_dim"]),
        )

        self.decoder_input = nn.Sequential(
            nn.Linear(config["latent_dim"] + ant_feat_dim, config["fusion_dim"]),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["fusion_dim"], config["latent_dim"]),
        )

        self.length_head = nn.Sequential(
            nn.Linear(config["latent_dim"], config["latent_dim"]),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["latent_dim"], config["max_cdr3_len"]),
        )

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


def encode_seq(seq, fixed_len):
    seq = seq[:fixed_len]
    ids = [stoi.get(ch, UNK_IDX) for ch in seq]
    if len(ids) < fixed_len:
        ids += [PAD_IDX] * (fixed_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


def decode_tokens(token_ids):
    chars = []
    for idx in token_ids:
        ch = itos[idx]
        if ch == "<PAD>":
            continue
        chars.append(ch)
    return "".join(chars)


@torch.no_grad()
def generate_from_antigen(
    model,
    antigen_seq,
    antigen_max_len,
    max_cdr3_len,
    num_samples,
    device,
    min_len=8,
    temperature=1.0,
    sample_mode="sample",
):
    model.eval()

    a = encode_seq(antigen_seq, antigen_max_len).unsqueeze(0).to(device)
    a_mask = (a != PAD_IDX).long()

    ha = model.antigen_encoder(a, a_mask)
    ha = ha.repeat(num_samples, 1)

    mu_p = model.prior_mu(ha)
    logvar_p = model.prior_logvar(ha)
    z = model.reparameterize(mu_p, logvar_p)

    z_cond = model.decoder_input(torch.cat([z, ha], dim=-1))
    logits = model.decoder(z_cond)
    len_logits = model.length_head(z_cond)

    if temperature != 1.0:
        logits = logits / temperature

    pred_lens = torch.argmax(len_logits, dim=-1) + 1
    pred_lens = torch.clamp(pred_lens, min=min_len, max=max_cdr3_len)

    if sample_mode == "sample":
        probs = torch.softmax(logits, dim=-1)
        preds = torch.multinomial(
            probs.reshape(-1, probs.size(-1)), num_samples=1
        ).view(num_samples, max_cdr3_len)
    else:
        preds = logits.argmax(dim=-1)

    seqs = []
    for i in range(num_samples):
        L = int(pred_lens[i].item())
        token_ids = preds[i][:L].cpu().tolist()
        seq = decode_tokens(token_ids)
        seqs.append((L, seq))

    return seqs


def main():
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    config = ckpt["config"]
    antigen_max_len = ckpt["max_antigen_len"]

    model = ConditionalCNNVAE(config).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    seqs = generate_from_antigen(
        model=model,
        antigen_seq=TARGET_ANTIGEN,
        antigen_max_len=antigen_max_len,
        max_cdr3_len=config["max_cdr3_len"],
        num_samples=NUM_SAMPLES,
        device=DEVICE,
        min_len=MIN_LEN,
        temperature=TEMPERATURE,
        sample_mode=SAMPLE_MODE,
    )

    print("Target antigen:")
    print(TARGET_ANTIGEN)
    print("\nGenerated CDRH3 sequences:\n")

    for i, (L, seq) in enumerate(seqs, 1):
        print(f"{i:02d}\tlen={L}\t{seq}")

    with open("generated_cdrh3_from_antigenfinetune.txt", "w") as f:
        f.write("Target antigen:\n")
        f.write(TARGET_ANTIGEN + "\n\n")
        f.write("Generated CDRH3 sequences:\n")
        for i, (L, seq) in enumerate(seqs, 1):
            f.write(f"{i:02d}\tlen={L}\t{seq}\n")


if __name__ == "__main__":
    main()
