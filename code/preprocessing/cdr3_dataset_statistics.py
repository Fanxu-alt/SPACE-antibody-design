
import json
import hashlib
from pathlib import Path
from collections import Counter
from importlib.metadata import version

import numpy as np
import pandas as pd


CSV_PATH = Path("CoV-AbDab.csv")
CDR3_COLUMN = "cdr3"
OUTPUT_DIR = Path("cdr3_statistics")

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)



def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(chunk)

    return h.hexdigest()


def validate_cdr3(value):
    if pd.isna(value):
        return None, "missing"

    seq = str(value).strip().upper()

    if not seq:
        return None, "empty"

    invalid = set(seq) - VALID_AA

    if invalid:
        return None, "invalid_amino_acid"

    return seq, "valid"


def describe_lengths(sequences):
    lengths = np.array(
        [len(s) for s in sequences],
        dtype=np.int64
    )

    if lengths.size == 0:
        raise ValueError(
            "No valid CDR3 sequences available."
        )

    return {
        "count": int(lengths.size),
        "mean": float(np.mean(lengths)),
        "std": float(np.std(lengths, ddof=1))
        if lengths.size > 1 else None,
        "median": float(np.median(lengths)),
        "min": int(np.min(lengths)),
        "max": int(np.max(lengths)),
        "q1": float(np.percentile(lengths, 25)),
        "q3": float(np.percentile(lengths, 75)),
    }

df = pd.read_csv(CSV_PATH, dtype={CDR3_COLUMN: "string"})

if CDR3_COLUMN not in df.columns:
    raise ValueError(
        f"Missing required column: {CDR3_COLUMN}"
    )

records = [
    validate_cdr3(x)
    for x in df[CDR3_COLUMN]
]

df["cdr3_clean"] = [x[0] for x in records]
df["qc_status"] = [x[1] for x in records]

valid_df = df.loc[
    df["qc_status"] == "valid"
].copy()

valid_sequences = valid_df[
    "cdr3_clean"
].tolist()

if not valid_sequences:
    raise ValueError(
        "No valid CDR3 sequences after filtering."
    )

unique_sequences = list(
    dict.fromkeys(valid_sequences)
)


all_stats = describe_lengths(
    valid_sequences
)

unique_stats = describe_lengths(
    unique_sequences
)

summary = {
    "input_file": str(CSV_PATH),
    "input_sha256": sha256_file(CSV_PATH),
    "cdr3_column": CDR3_COLUMN,
    "total_input_rows": int(len(df)),
    "valid_rows": int(len(valid_sequences)),
    "unique_valid_sequences": int(
        len(unique_sequences)
    ),
    "duplicate_valid_rows": int(
        len(valid_sequences)
        - len(unique_sequences)
    ),
    "valid_fraction": float(
        len(valid_sequences) / len(df)
    ) if len(df) else None,
    "qc_counts": {
        str(k): int(v)
        for k, v in df[
            "qc_status"
        ].value_counts().items()
    },
    "all_valid_length_stats": all_stats,
    "unique_length_stats": unique_stats,
    "numpy_version": version("numpy"),
    "pandas_version": version("pandas"),
}



def length_distribution(sequences):
    counts = Counter(
        len(seq) for seq in sequences
    )

    total = len(sequences)

    return pd.DataFrame([
        {
            "length": length,
            "count": count,
            "frequency": count / total
        }
        for length, count in sorted(
            counts.items()
        )
    ])


length_all = length_distribution(
    valid_sequences
)

length_unique = length_distribution(
    unique_sequences
)



def amino_acid_composition(sequences):
    counter = Counter()

    for seq in sequences:
        counter.update(seq)

    total = sum(counter.values())

    return pd.DataFrame([
        {
            "amino_acid": aa,
            "count": counter[aa],
            "frequency": counter[aa] / total,
        }
        for aa in sorted(VALID_AA)
    ])


aa_all = amino_acid_composition(
    valid_sequences
)

aa_unique = amino_acid_composition(
    unique_sequences
)




df.to_csv(
    OUTPUT_DIR / "cdr3_qc_records.csv",
    index=False
)

valid_df.to_csv(
    OUTPUT_DIR / "cdr3_valid_records.csv",
    index=False
)

pd.DataFrame({
    "cdr3": unique_sequences
}).to_csv(
    OUTPUT_DIR / "cdr3_unique_sequences.csv",
    index=False
)

length_all.to_csv(
    OUTPUT_DIR / "cdr3_length_all.csv",
    index=False
)

length_unique.to_csv(
    OUTPUT_DIR / "cdr3_length_unique.csv",
    index=False
)

aa_all.to_csv(
    OUTPUT_DIR / "cdr3_aa_frequency_all.csv",
    index=False
)

aa_unique.to_csv(
    OUTPUT_DIR / "cdr3_aa_frequency_unique.csv",
    index=False
)

with open(
    OUTPUT_DIR / "cdr3_summary.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        summary,
        f,
        indent=2,
        ensure_ascii=False
    )



print("=" * 60)
print("CoV-AbDab CDR3 Sequence Statistics")
print("=" * 60)

print(
    "Input rows:",
    summary["total_input_rows"]
)

print(
    "Valid sequences:",
    summary["valid_rows"]
)

print(
    "Unique sequences:",
    summary["unique_valid_sequences"]
)

print(
    "Duplicate records:",
    summary["duplicate_valid_rows"]
)

print("\nLength statistics (all valid records):")

for key, value in all_stats.items():
    print(f"{key}: {value}")

print("\nLength statistics (unique sequences):")

for key, value in unique_stats.items():
    print(f"{key}: {value}")

print("\nQuality control:")

for key, value in summary["qc_counts"].items():
    print(f"{key}: {value}")

print("\nOutputs saved to:", OUTPUT_DIR)
