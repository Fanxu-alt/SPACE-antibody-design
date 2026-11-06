#!/usr/bin/env python3

import argparse
import csv
import gzip
import hashlib
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

VALID_AA = frozenset('ACDEFGHIKLMNPQRSTVWY')

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def process_file(path, conn, batch_size, min_len, max_len):
    stats = Counter()
    pending = []
    with gzip.open(path, 'rt', encoding='utf-8-sig', newline='') as f:
        metadata_line = f.readline()
        if not metadata_line:
            raise ValueError('empty gzip member')
        try:
            metadata = json.loads(metadata_line)
            if not isinstance(metadata, dict):
                raise ValueError('metadata must be a JSON object')
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError('first line is not valid OAS JSON metadata') from exc
        reader = csv.DictReader(f, strict=True)
        if not reader.fieldnames or 'cdr3_aa' not in reader.fieldnames:
            raise ValueError('missing cdr3_aa column')
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError('duplicate CSV header columns')
        for row in reader:
            stats['input_rows'] += 1
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f'malformed CSV record near line {reader.line_num}')
            value = row['cdr3_aa']
            seq = value.strip().upper()
            if not seq:
                stats['missing_or_empty'] += 1
            elif not set(seq) <= VALID_AA:
                stats['noncanonical'] += 1
            elif min_len is not None and len(seq) < min_len:
                stats['below_min_length'] += 1
            elif max_len is not None and len(seq) > max_len:
                stats['above_max_length'] += 1
            else:
                stats['valid_rows'] += 1
                pending.append((seq,))
                if len(pending) >= batch_size:
                    conn.executemany('INSERT OR IGNORE INTO seqs(seq) VALUES (?)', pending)
                    pending.clear()
        if pending:
            conn.executemany('INSERT OR IGNORE INTO seqs(seq) VALUES (?)', pending)
    return {'file': str(path), 'sha256': sha256(path), 'metadata': metadata, 'counts': dict(stats)}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', type=Path, default=Path('.'))
    p.add_argument('--output-dir', type=Path, default=Path('cleaned'))
    p.add_argument('--batch-size', type=int, default=10000)
    p.add_argument('--min-length', type=int, default=None)
    p.add_argument('--max-length', type=int, default=None)
    args = p.parse_args()
    if args.batch_size < 1:
        p.error('--batch-size must be positive')
    if args.min_length is not None and args.min_length < 1:
        p.error('--min-length must be positive')
    if args.max_length is not None and args.max_length < 1:
        p.error('--max-length must be positive')
    if args.min_length is not None and args.max_length is not None and args.min_length > args.max_length:
        p.error('min-length exceeds max-length')
    paths = sorted(args.input_dir.glob('*.csv.gz'))
    if not paths:
        p.error('No *.csv.gz files found')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    db_path = args.output_dir / 'oas_cdrh3_dedup.sqlite'
    if db_path.exists():
        p.error(f'Database already exists: {db_path} (remove it before a fresh run)')
    output = args.output_dir / 'covid_human_heavy_cdr3_aa_unique.txt'
    manifest = args.output_dir / 'oas_processing_manifest.json'
    if output.exists() or manifest.exists():
        p.error('Output or manifest already exists; use a fresh output directory')
    conn = sqlite3.connect(db_path)
    summaries = []
    try:
        conn.execute('CREATE TABLE seqs (seq TEXT PRIMARY KEY) WITHOUT ROWID')
        for idx, path in enumerate(paths, 1):
            print(f'[{idx}/{len(paths)}] {path.name}', flush=True)
            # Per-file transaction: invalid/corrupt files never contribute partial data.
            try:
                conn.execute('BEGIN')
                report = process_file(path, conn, args.batch_size, args.min_length, args.max_length)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            summaries.append(report)
        unique_count = conn.execute('SELECT COUNT(*) FROM seqs').fetchone()[0]
        tmp = output.with_suffix(output.suffix + '.partial')
        with tmp.open('w', encoding='ascii', newline='\n') as f:
            for (seq,) in conn.execute('SELECT seq FROM seqs ORDER BY seq'):
                f.write(seq + '\n')
        tmp.replace(output)
        totals = Counter()
        for r in summaries:
            totals.update(r['counts'])
        report = {
            'input_directory': str(args.input_dir.resolve()),
            'input_file_count': len(paths),
            'cdr3_field': 'cdr3_aa',
            'allowed_amino_acids': ''.join(sorted(VALID_AA)),
            'min_length': args.min_length,
            'max_length': args.max_length,
            'deduplication': 'exact amino acid sequence globally across all files',
            'output_sha256': sha256(output),
            'unique_sequences': unique_count,
            'total_counts': dict(totals),
            'input_files': summaries,
        }
        manifest.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f'Valid rows: {totals["valid_rows"]:,}')
        print(f'Unique sequences: {unique_count:,}')
        print(f'Output: {output}')
        print(f'Manifest: {manifest}')
    finally:
        conn.close()

if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}', file=sys.stderr)
        sys.exit(1)
