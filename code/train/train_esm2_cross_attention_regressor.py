#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import GroupShuffleSplit
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

CANONICAL = frozenset('ACDEFGHIKLMNPQRSTVWY')


@dataclass
class Config:
    csv_path: str = 'biomap.csv'
    output_dir: str = 'affinity_five_run_results'
    heavy_col: str = 'ab'
    antigen_col: str = 'ag'
    label_col: str = 'g'
    model_name: str = 'facebook/esm2_t6_8M_UR50D'
    model_revision: str = 'main'  
    max_heavy_len: int = 256  
    max_antigen_len: int = 512
    allow_truncation: bool = False
    batch_size: int = 8
    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-5
    num_heads: int = 8
    hidden_dim: int = 256
    dropout: float = 0.1
    num_workers: int = 0
    seed_list: tuple[int, ...] = (42, 43, 44, 45, 46)
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
   
    patience: int = 0  
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for piece in iter(lambda: f.read(1024 * 1024), b''):
            h.update(piece)
    return h.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate_data(path: Path, cfg: Config) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, dtype={cfg.heavy_col: 'string', cfg.antigen_col: 'string'})
    required = [cfg.heavy_col, cfg.antigen_col, cfg.label_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'{path}: missing columns {missing}')
    if df.empty:
        raise ValueError(f'{path}: empty input')
    df = df[required].copy()
    for col in (cfg.heavy_col, cfg.antigen_col):
        if df[col].isna().any():
            raise ValueError(f'{path}: missing {col} sequence')
        df[col] = df[col].str.strip().str.upper()
        bad = df[col].map(lambda s: not s or bool(set(s) - CANONICAL))
        if bad.any():
            raise ValueError(f'{path}: invalid {col} at rows {np.flatnonzero(bad.to_numpy())[:8].tolist()}')
    values = pd.to_numeric(df[cfg.label_col], errors='coerce')
    if values.isna().any() or not np.isfinite(values.astype(float)).all():
        raise ValueError(f'{path}: regression targets must be finite numeric values')
    df[cfg.label_col] = values.astype(float)
    df.insert(0, 'source_row_index', range(len(df)))
    return df.reset_index(drop=True)


def split_dataset(df: pd.DataFrame, cfg: Config, seed: int):
    """Pair-group-disjoint split. Approximate ratios due to variable group sizes."""
    groups = (df[cfg.heavy_col] + '|' + df[cfg.antigen_col]).to_numpy()
    if len(np.unique(groups)) < 3:
        raise ValueError('Need at least 3 distinct sequence pairs for grouped train/val/test splitting')
    indices = np.arange(len(df))
    first = GroupShuffleSplit(n_splits=1, test_size=cfg.test_ratio, random_state=seed)
    train_val, test = next(first.split(indices, groups=groups))
    second = GroupShuffleSplit(n_splits=1,
                               test_size=cfg.val_ratio / (cfg.train_ratio + cfg.val_ratio),
                               random_state=seed + 10000)
    train_sub, val_sub = next(second.split(train_val, groups=groups[train_val]))
    train, val = train_val[train_sub], train_val[val_sub]
    if not all(len(x) for x in (train, val, test)):
        raise ValueError('Empty train/val/test split')
    sets = [set(groups[idx]) for idx in (train, val, test)]
    if any(sets[i] & sets[j] for i, j in ((0,1), (0,2), (1,2))):
        raise RuntimeError('Exact pair leakage across splits')
    return {'train': train, 'val': val, 'test': test}


class PairDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg: Config):
        self.records = list(zip(df[cfg.heavy_col], df[cfg.antigen_col], df[cfg.label_col], df['source_row_index']))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        h, a, y, row = self.records[idx]
        return {'heavy': h, 'antigen': a, 'label': float(y), 'row': int(row)}


class PairCollator:
    def __init__(self, tokenizer, cfg: Config):
        self.tokenizer = tokenizer
        self.cfg = cfg
        special = tokenizer.num_special_tokens_to_add(pair=False)
        self.capacity_h = cfg.max_heavy_len - special
        self.capacity_a = cfg.max_antigen_len - special
        if self.capacity_h < 1 or self.capacity_a < 1:
            raise ValueError('Maximum token lengths must exceed special-token count')

    def __call__(self, batch):
        hs = [x['heavy'] for x in batch]
        ags = [x['antigen'] for x in batch]
        if not self.cfg.allow_truncation:
            if any(len(s) > self.capacity_h for s in hs):
                raise ValueError('Heavy exceeds max_heavy_len; increase limit or explicitly allow truncation')
            if any(len(s) > self.capacity_a for s in ags):
                raise ValueError('Antigen exceeds max_antigen_len; increase limit or explicitly allow truncation')
        def tok(seqs, maximum):
            return self.tokenizer([' '.join(s) for s in seqs], padding=True,
                                  truncation=self.cfg.allow_truncation, max_length=maximum,
                                  return_special_tokens_mask=True, return_tensors='pt')
        h, a = tok(hs, self.cfg.max_heavy_len), tok(ags, self.cfg.max_antigen_len)
        return {'h_ids': h['input_ids'], 'h_mask': h['attention_mask'],
                'h_res': h['attention_mask'] * (1 - h['special_tokens_mask']),
                'a_ids': a['input_ids'], 'a_mask': a['attention_mask'],
                'a_res': a['attention_mask'] * (1 - a['special_tokens_mask']),
                'labels': torch.tensor([x['label'] for x in batch], dtype=torch.float32),
                'rows': [x['row'] for x in batch]}


def masked_mean(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    mask = valid.unsqueeze(-1).to(x.dtype)
    return (x * mask).sum(1) / mask.sum(1).clamp_min(1.0)


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, kv, kv_mask):
        out, _ = self.attn(q, kv, kv, key_padding_mask=~kv_mask.bool(), need_weights=False)
        return self.norm(q + self.dropout(out))


class AbAgAffinityRegressor(nn.Module):
    def __init__(self, cfg: Config, esm=None):
        super().__init__()
        if esm is None:
            from transformers import AutoModel
            esm = AutoModel.from_pretrained(cfg.model_name, revision=cfg.model_revision)
        self.esm = esm
        self.esm.requires_grad_(False)
        d = self.esm.config.hidden_size
        if d % cfg.num_heads:
            raise ValueError(f'ESM hidden size {d} not divisible by attention heads {cfg.num_heads}')
        self.ab_to_ag = CrossAttentionBlock(d, cfg.num_heads, cfg.dropout)
        self.ag_to_ab = CrossAttentionBlock(d, cfg.num_heads, cfg.dropout)
        self.ab_proj = nn.Linear(d, cfg.hidden_dim)
        self.ag_proj = nn.Linear(d, cfg.hidden_dim)
        self.regressor = nn.Sequential(nn.Linear(4 * cfg.hidden_dim, cfg.hidden_dim),
                                        nn.ReLU(), nn.Dropout(cfg.dropout), nn.Linear(cfg.hidden_dim, 1))

    def train(self, mode: bool = True):
        super().train(mode)
        self.esm.eval()  
        return self

    def forward(self, h_ids, h_mask, h_res, a_ids, a_mask, a_res):
        with torch.no_grad():
            h = self.esm(input_ids=h_ids, attention_mask=h_mask).last_hidden_state
            a = self.esm(input_ids=a_ids, attention_mask=a_mask).last_hidden_state
        hc = self.ab_to_ag(h, a, a_res)
        ac = self.ag_to_ab(a, h, h_res)
        hv = masked_mean(self.ab_proj(hc), h_res)
        av = masked_mean(self.ag_proj(ac), a_res)
        features = torch.cat([hv, av, torch.abs(hv - av), hv * av], dim=-1)
        return self.regressor(features).squeeze(-1)


def evaluate_metrics(y, p) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise RuntimeError('Non-finite labels/predictions')
    mse = float(np.mean((y-p)**2))
    mae = float(np.mean(np.abs(y-p)))
    pearson = float(pearsonr(y, p).statistic) if len(y) >= 2 and np.std(y) > 1e-12 and np.std(p) > 1e-12 else float('nan')
    spearman = float(spearmanr(y, p).statistic) if len(y) >= 2 and np.std(y) > 1e-12 and np.std(p) > 1e-12 else float('nan')
    return {'Pearson': pearson, 'Spearman': spearman, 'MSE': mse, 'RMSE': float(np.sqrt(mse)), 'MAE': mae}


def run_epoch(model, loader, device, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, total_n = 0.0, 0
    ys, ps, indices = [], [], []
    for batch in loader:
        features = {k: batch[k].to(device) for k in ('h_ids', 'h_mask', 'h_res', 'a_ids', 'a_mask', 'a_res')}
        labels = batch['labels'].to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            preds = model(**features)
            loss = criterion(preds, labels)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss')
            if training:
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item()) * len(labels)
        total_n += len(labels)
        ys.extend(labels.detach().cpu().tolist())
        ps.extend(preds.detach().cpu().tolist())
        indices.extend(batch['rows'])
    if not total_n:
        raise ValueError('Empty DataLoader')
    return total_loss / total_n, evaluate_metrics(ys, ps), pd.DataFrame(
        {'source_row_index': indices, 'target': ys, 'prediction': ps})


def trainable_state(model: nn.Module) -> dict:
 
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if not k.startswith('esm.')}


def restore_trainable_state(model: nn.Module, state: dict) -> None:
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys or any(not k.startswith('esm.') for k in result.missing_keys):
        raise RuntimeError(f'Incorrect checkpoint parameters: {result}')


def get_versions():
    result = {'python': sys.version.split()[0], 'torch': torch.__version__,
              'numpy': np.__version__, 'pandas': pd.__version__}
    for name in ('transformers', 'scikit-learn', 'scipy'):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = 'not-installed'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, field in Config.__dataclass_fields__.items():
        default = field.default
        key = '--' + name.replace('_', '-')
        if name == 'seed_list':
            parser.add_argument(key, nargs='+', type=int, default=list(default))
        elif isinstance(default, bool):
            parser.add_argument(key, action='store_true', default=default)
        else:
            parser.add_argument(key, type=type(default), default=default)
    args = parser.parse_args()
    cfg = Config(**{k: tuple(v) if k == 'seed_list' else v for k, v in vars(args).items()})
    if cfg.batch_size < 1 or cfg.epochs < 1 or cfg.num_heads < 1 or not cfg.seed_list:
        raise ValueError('Invalid training hyperparameters')
    if min(cfg.train_ratio, cfg.val_ratio, cfg.test_ratio) <= 0 or not np.isclose(cfg.train_ratio+cfg.val_ratio+cfg.test_ratio, 1.0):
        raise ValueError('Train/val/test ratios must be positive and sum to 1')
    if cfg.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, revision=cfg.model_revision)
    collator = PairCollator(tokenizer, cfg)
    df = validate_data(Path(cfg.csv_path), cfg)
    if not cfg.allow_truncation:
        if (df[cfg.heavy_col].str.len() > collator.capacity_h).any() or (df[cfg.antigen_col].str.len() > collator.capacity_a).any():
            raise ValueError('Sequences exceed ESM token limits. Choose an explicit truncation policy or increase limits.')
    outdir = Path(cfg.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir/'config.json').write_text(json.dumps(asdict(cfg), indent=2))
    (outdir/'environment.json').write_text(json.dumps(get_versions(), indent=2))
    (outdir/'dataset_manifest.json').write_text(json.dumps({'input_sha256': file_sha256(Path(cfg.csv_path)), 'n_rows': len(df),
       'n_unique_pairs': int(df[[cfg.heavy_col,cfg.antigen_col]].drop_duplicates().shape[0]),
       'target_mean': float(df[cfg.label_col].mean()), 'target_std': float(df[cfg.label_col].std())}, indent=2))
    criterion = nn.MSELoss()
    summary = []
    for run, seed in enumerate(cfg.seed_list, start=1):
        set_seed(seed)
        rundir = outdir/f'run{run}_seed{seed}'
        rundir.mkdir(exist_ok=True)
        indices = split_dataset(df, cfg, seed)
        split_rows = []
        for name, subset in indices.items():
            for idx in subset:
                split_rows.append({'source_row_index': int(idx), 'split': name})
        pd.DataFrame(split_rows).sort_values('source_row_index').to_csv(rundir/'split_assignments.csv', index=False)
        audit = {name: {'rows': int(len(ids)), 'unique_pairs': int(df.iloc[ids][[cfg.heavy_col,cfg.antigen_col]].drop_duplicates().shape[0]),
                        'target_mean': float(df.iloc[ids][cfg.label_col].mean())} for name, ids in indices.items()}
        audit['split_type'] = 'pair-group-disjoint; antibody/antigen sequences may overlap'
        (rundir/'split_audit.json').write_text(json.dumps(audit, indent=2))
        generator = torch.Generator().manual_seed(seed)
        loaders = {name: DataLoader(PairDataset(df.iloc[ids], cfg), batch_size=cfg.batch_size,
                                   shuffle=name=='train', generator=generator if name=='train' else None,
                                   num_workers=cfg.num_workers, collate_fn=collator)
                   for name, ids in indices.items()}
        model = AbAgAffinityRegressor(cfg).to(cfg.device)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=cfg.lr, weight_decay=cfg.weight_decay)
      
        best_pearson, best_epoch, stale = -float('inf'), -1, 0
        history = []
        ckpt_path = rundir/'best_trainable.pt'
        for epoch in range(1, cfg.epochs+1):
            tr_loss, tr_metrics, _ = run_epoch(model, loaders['train'], cfg.device, criterion, optimizer)
            with torch.no_grad():
                va_loss, va_metrics, _ = run_epoch(model, loaders['val'], cfg.device, criterion)
            history.append({'epoch': epoch, 'train_loss': tr_loss, 'val_loss': va_loss,
                **{f'train_{k}': v for k,v in tr_metrics.items()}, **{f'val_{k}': v for k,v in va_metrics.items()}})
            pd.DataFrame(history).to_csv(rundir/'training_history.csv', index=False)
            score = va_metrics['Pearson']
            print(f'run={run} seed={seed} epoch={epoch} val_Pearson={score:.4f} val_MSE={va_loss:.4f}', flush=True)
            if np.isfinite(score) and score > best_pearson:
                best_pearson, best_epoch, stale = score, epoch, 0
                torch.save({'trainable_state_dict': trainable_state(model), 'model_name': cfg.model_name,
                   'model_revision': cfg.model_revision, 'epoch': epoch, 'seed': seed, 'config': asdict(cfg)}, ckpt_path)
            else:
                stale += 1
            if cfg.patience and stale >= cfg.patience:
                break
        if best_epoch < 0:
            raise RuntimeError('No valid validation Pearson; check validation size, variation and model predictions')
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        restore_trainable_state(model, checkpoint['trainable_state_dict'])
        with torch.no_grad():
            test_loss, test_metrics, test_preds = run_epoch(model, loaders['test'], cfg.device, criterion)
        test_preds.sort_values('source_row_index').to_csv(rundir/'test_predictions.csv', index=False)
        result = {'Run': run, 'Seed': seed, 'Best_epoch': best_epoch, 'Best_val_Pearson': best_pearson,
                  'Test_loss': test_loss, **test_metrics}
        summary.append(result)
        pd.DataFrame(summary).to_csv(outdir/'five_run_test_results.csv', index=False)
        print('Held-out test:', result, flush=True)
        del model, optimizer
        if cfg.device == 'cuda':
            torch.cuda.empty_cache()
    res = pd.DataFrame(summary)
    metrics = ('Pearson','Spearman','MSE','RMSE','MAE')
    aggregate = [{'Metric': metric, 'Mean': float(res[metric].mean()),
                  'SD': float(res[metric].std(ddof=1)) if len(res)>1 else None} for metric in metrics]
    pd.DataFrame(aggregate).to_csv(outdir/'five_run_summary.csv', index=False)
    print(pd.DataFrame(aggregate).to_string(index=False))

if __name__ == '__main__':
    main()
