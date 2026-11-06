
import json
import hashlib
import platform
from pathlib import Path
from importlib.metadata import version

import pandas as pd
from anarcii import Anarcii


INPUT_FILE = Path("sabdab_positive_heavy_antigen.csv")
OUTPUT_FILE = Path("sabdab_positive_heavy_antigen_cdrh3_imgt.csv")
BATCH_SIZE = 128
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_sequence(value):
    if pd.isna(value):
        return None, "missing_sequence"

    seq = str(value).strip().upper()

    if not seq:
        return None, "empty_sequence"

    if any(aa not in VALID_AA for aa in seq):
        return None, "invalid_amino_acid"

    return seq, None


def extract_cdrh3(item):
    if item is None:
        return None, None, None, "missing_result"

    if item.get("error") is not None:
        return None, None, None, (
            "anarcii_error:" + str(item["error"])
        )

    if item.get("chain_type") != "H":
        return None, None, None, "not_heavy_chain"

    if item.get("scheme") != "imgt":
        return None, None, None, "incorrect_scheme"

    numbering = item.get("numbering")

    if not numbering:
        return None, None, None, "missing_numbering"

    residues = []
    positions = {}
    cdr_numbering = []

    for (pos, ins), aa in numbering:
        ins = str(ins).strip()
        label = f"{pos}{ins}"

        if aa != "-":
            positions[pos] = aa

        if 105 <= pos <= 117 and aa != "-":
            residues.append(aa)
            cdr_numbering.append([label, aa])

    if not residues:
        return None, None, None, "empty_cdrh3"

  
    if positions.get(104) != "C":
        return None, None, None, "invalid_anchor_104"

    if positions.get(118) not in {"W", "F"}:
        return None, None, None, "invalid_anchor_118"

    cdrh3 = "".join(residues)

    if any(aa not in VALID_AA for aa in cdrh3):
        return None, None, None, "invalid_cdrh3"

    return (
        cdrh3,
        json.dumps(cdr_numbering),
        len(cdrh3),
        "success",
    )


def main():
    df = pd.read_csv(INPUT_FILE)

    if "heavy_seq" not in df.columns:
        raise ValueError("Missing column: heavy_seq")

    df = df.copy()
    df["source_row_id"] = range(len(df))

    validated = [
        validate_sequence(seq)
        for seq in df["heavy_seq"]
    ]

    sequences = [x[0] for x in validated]
    errors = [x[1] for x in validated]

    results = [
        (None, None, None, error)
        if error is not None
        else None
        for error in errors
    ]

    model = Anarcii(seq_type="antibody")

    valid_indices = [
        i for i, seq in enumerate(sequences)
        if seq is not None
    ]

    for start in range(0, len(valid_indices), BATCH_SIZE):
        batch_ids = valid_indices[start:start + BATCH_SIZE]
        batch_seqs = [sequences[i] for i in batch_ids]

        try:
            batch_results = model.number(batch_seqs)

            for j, original_idx in enumerate(batch_ids):
                key = f"Sequence {j + 1}"

                if key not in batch_results:
                    raise RuntimeError(
                        f"Missing ANARCII output: {key}"
                    )

                results[original_idx] = extract_cdrh3(
                    batch_results[key]
                )

        except Exception as exc:
            raise RuntimeError(
                f"ANARCII batch failed at offset {start}"
            ) from exc

        print(
            f"Processed {min(start + BATCH_SIZE, len(valid_indices))}"
            f"/{len(valid_indices)} valid sequences"
        )

    (
        df["cdrh3_seq"],
        df["cdrh3_imgt_numbering"],
        df["cdrh3_len"],
        df["extraction_status"],
    ) = zip(*results) if results else ([], [], [], [])

    df["cdrh3_len"] = pd.array(
        df["cdrh3_len"], dtype="Int64"
    )

    df.to_csv(OUTPUT_FILE, index=False)

    successful = df["extraction_status"] == "success"

    df.loc[successful].to_csv(
        OUTPUT_FILE.with_name(
            OUTPUT_FILE.stem + "_success.csv"
        ),
        index=False,
    )

    metadata = {
        "input_file": str(INPUT_FILE),
        "input_sha256": sha256_file(INPUT_FILE),
        "output_file": str(OUTPUT_FILE),
        "numbering_scheme": "IMGT",
        "cdrh3_positions": [105, 117],
        "anchor_validation": True,
        "batch_size": BATCH_SIZE,
        "anarcii_version": version("anarcii"),
        "pandas_version": pd.__version__,
        "python_version": platform.python_version(),
        "total_sequences": len(df),
        "success_count": int(successful.sum()),
        "failure_count": int((~successful).sum()),
        "status_counts": {
            str(k): int(v)
            for k, v in df["extraction_status"]
            .value_counts().items()
        },
    }

    if len(df):
        metadata["success_rate"] = float(successful.mean())
    else:
        metadata["success_rate"] = None

    metadata_path = OUTPUT_FILE.with_suffix(
        ".metadata.json"
    )

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
