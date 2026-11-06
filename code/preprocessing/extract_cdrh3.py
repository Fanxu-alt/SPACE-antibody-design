
import json
import hashlib
import platform
from pathlib import Path
from collections import Counter
from importlib.metadata import version, PackageNotFoundError

import pandas as pd
from anarci import anarci


INPUT_FILE = Path("heavy.csv")
OUTPUT_FILE = Path("heavy_with_cdrh3.csv")

SEQ_COLUMN = "heavy"
BATCH_SIZE = 100

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

IMGT_CDRH3_START = 105
IMGT_CDRH3_END = 117



def validate_sequence(value):
    if pd.isna(value):
        return None, "missing_sequence"

    seq = str(value).strip().upper()

    if not seq:
        return None, "empty_sequence"

    invalid = set(seq) - VALID_AA

    if invalid:
        return None, "invalid_amino_acid"

    return seq, "valid"




def extract_from_numbering(numbering, details):
    if details.get("chain_type") != "H":
        return None, None, "not_heavy_chain"

    if not numbering:
        return None, None, "empty_numbering"

    cdrh3 = []
    imgt_positions = []

    for (position, insertion), aa in numbering:
        if (
            IMGT_CDRH3_START <= position <= IMGT_CDRH3_END
            and aa != "-"
        ):
            cdrh3.append(aa)

            insertion = str(insertion).strip()
            imgt_positions.append({
                "position": position,
                "insertion": insertion,
                "amino_acid": aa
            })

    if not cdrh3:
        return None, None, "empty_cdrh3"

    sequence = "".join(cdrh3)

    if set(sequence) - VALID_AA:
        return None, None, "invalid_cdrh3"

    return (
        sequence,
        json.dumps(imgt_positions),
        "success"
    )


def process_batch(sequences):
 
    named_sequences = [
        (f"seq_{i}", seq)
        for i, seq in enumerate(sequences)
    ]

    numbering, details, _ = anarci(
        named_sequences,
        scheme="imgt",
        allow={"H"},
        output=False
    )

    if len(numbering) != len(sequences):
        raise RuntimeError(
            "ANARCI returned an unexpected number of results."
        )

    results = []

    for i in range(len(sequences)):
        domains = numbering[i]
        domain_details = details[i]

        if not domains:
            results.append(
                (None, None, "no_domain_detected")
            )
            continue

        if not domain_details:
            results.append(
                (None, None, "missing_alignment_details")
            )
            continue

        if len(domains) != 1:
            results.append(
                (None, None, "multiple_domains")
            )
            continue

        if len(domain_details) != 1:
            results.append(
                (None, None, "multiple_alignment_details")
            )
            continue

        numbered_residues = domains[0][0]

        results.append(
            extract_from_numbering(
                numbered_residues,
                domain_details[0]
            )
        )

    return results



def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def get_anarci_version():
    try:
        return version("anarci")
    except PackageNotFoundError:
        return "not_available"



def main():
    df = pd.read_csv(
        INPUT_FILE,
        dtype={SEQ_COLUMN: "string"}
    )

    if SEQ_COLUMN not in df.columns:
        raise ValueError(
            f"Missing required column: {SEQ_COLUMN}"
        )

    df = df.copy()

    df["source_row_id"] = range(len(df))

    validated = [
        validate_sequence(seq)
        for seq in df[SEQ_COLUMN]
    ]

    df["heavy_validated"] = [
        item[0] for item in validated
    ]

    df["input_qc_status"] = [
        item[1] for item in validated
    ]

    results = [
        (None, None, status)
        if status != "valid"
        else None
        for _, status in validated
    ]

    valid_indices = [
        i for i, (_, status) in enumerate(validated)
        if status == "valid"
    ]

    print("Total input rows:", len(df))
    print("Valid input sequences:", len(valid_indices))

    for start in range(
        0, len(valid_indices), BATCH_SIZE
    ):
        batch_indices = valid_indices[
            start:start + BATCH_SIZE
        ]

        batch_sequences = [
            df.iloc[i]["heavy_validated"]
            for i in batch_indices
        ]

        try:
            batch_results = process_batch(
                batch_sequences
            )
        except Exception as exc:
            raise RuntimeError(
                f"ANARCI failed at batch offset {start}"
            ) from exc

        for original_idx, result in zip(
            batch_indices,
            batch_results
        ):
            results[original_idx] = result

        processed = min(
            start + BATCH_SIZE,
            len(valid_indices)
        )

        print(
            f"Processed {processed}/"
            f"{len(valid_indices)}"
        )

    if results:
        (
            df["CDRH3"],
            df["cdrh3_imgt_numbering"],
            df["extraction_status"]
        ) = zip(*results)
    else:
        df["CDRH3"] = pd.Series(dtype="string")
        df["cdrh3_imgt_numbering"] = pd.Series(
            dtype="string"
        )
        df["extraction_status"] = pd.Series(
            dtype="string"
        )

    df["cdrh3_len"] = (
        df["CDRH3"]
        .str.len()
        .astype("Int64")
    )

 
    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    success_mask = (
        df["extraction_status"] == "success"
    )

    success_df = df.loc[success_mask].copy()

    success_file = OUTPUT_FILE.with_name(
        "heavy_with_cdrh3_success.csv"
    )

    success_df.to_csv(
        success_file,
        index=False
    )

    status_counts = Counter(
        df["extraction_status"].tolist()
    )

    metadata = {
        "input_file": str(INPUT_FILE),
        "input_sha256": sha256_file(INPUT_FILE),
        "output_file": str(OUTPUT_FILE),
        "sequence_column": SEQ_COLUMN,
        "numbering_tool": "ANARCI",
        "anarci_version": get_anarci_version(),
        "numbering_scheme": "IMGT",
        "chain_type": "H",
        "cdrh3_start": IMGT_CDRH3_START,
        "cdrh3_end": IMGT_CDRH3_END,
        "batch_size": BATCH_SIZE,
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "total_rows": len(df),
        "valid_input_sequences": len(valid_indices),
        "successful_extractions": len(success_df),
        "failed_extractions": len(df) - len(success_df),
        "success_rate": (
            len(success_df) / len(df)
            if len(df) else None
        ),
        "extraction_status_counts": dict(
            status_counts
        ),
    }

    metadata_file = OUTPUT_FILE.with_suffix(
        ".metadata.json"
    )

    with metadata_file.open(
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\n===== EXTRACTION SUMMARY =====")
    print("Input:", len(df))
    print("Successful:", len(success_df))
    print("Failed:", len(df) - len(success_df))

    for status, count in sorted(
        status_counts.items()
    ):
        print(f"{status}: {count}")

    print("\nSaved:", OUTPUT_FILE)
    print("Saved:", success_file)
    print("Metadata:", metadata_file)


if __name__ == "__main__":
    main()
