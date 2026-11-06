from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import platform
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOG = logging.getLogger("space.developability")
AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
KD = dict(zip("IVLFCMAGTSWYPHEQDNKR", [4.5, 4.2, 3.8, 2.8, 2.5, 1.9, 1.8, -0.4, -0.7, -0.8, -0.9, -1.3, -1.6, -3.2, -3.5, -3.5, -3.5, -3.5, -3.9, -4.5]))
PKA = {"Nterm": 8.0, "Cterm": 3.1, "C": 8.5, "D": 3.9, "E": 4.1, "H": 6.0, "K": 10.5, "R": 12.5, "Y": 10.1}
POS, NEG = frozenset("KRH"), frozenset("DE")
HYDROPHOBIC, AROMATIC = frozenset("AILMFWVYC"), frozenset("FWY")

DEFAULT_CONFIG: dict[str, Any] = {
    "pH": 7.4,
    "thresholds": {
        "cdr3_len_min": 8, "cdr3_len_max": 20,
        "cdr3_cysteines_max": 0, "cdr3_n_glyco_motifs_max": 0,
        "cdr3_deamidation_motifs_max": 1, "cdr3_isomerization_motifs_max": 1,
        "cdr3_hydrophobic_run_max": 5, "cdr3_gravy_max": 0.8,
        "cdr3_abs_charge_max": 4.0, "cdr3_methionines_max": 1,
    },
    "motif_weights": {
        "cdr3_n_glyco_motifs": 4.0, "cdr3_deamidation_motifs": 2.0,
        "cdr3_isomerization_motifs": 2.0, "cdr3_cysteines": 4.0,
        "cdr3_methionines": 0.75,
    },
    "upper_tail_metrics": ["cdr3_gravy", "cdr3_hydrophobic_run", "cdr3_max_window_gravy_5", "cdr3_max_window_abs_charge_5"],
    "two_sided_metrics": ["cdr3_len"],
    "charge_tolerance": 2.5,
    "charge_penalty_weight": 0.75,
    "risk_percentile_max": 50.0,
    "heavy_novelty_min": 0.10,
    "cdr3_novelty_min": 0.20,
    "require_cdr3_in_heavy": True,
}


def validate_config(cfg: dict[str, Any]) -> None:
    """Avoid silently changing scientific definitions through partial configs."""
    if set(cfg) != set(DEFAULT_CONFIG):
        raise ValueError("Config keys must match defaults exactly")
    for key in ("thresholds", "motif_weights"):
        if set(cfg[key]) != set(DEFAULT_CONFIG[key]):
            raise ValueError(f"Config {key} keys must match defaults exactly")
    for k, v in cfg["thresholds"].items():
        if not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"Invalid threshold {k}={v}")
    if cfg["thresholds"]["cdr3_len_min"] > cfg["thresholds"]["cdr3_len_max"]:
        raise ValueError("Minimum CDR3 length exceeds maximum")
    for key in ("motif_weights",):
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in cfg[key].values()):
            raise ValueError(f"All {key} must be nonnegative finite numbers")
    for key in ("pH", "charge_tolerance", "charge_penalty_weight", "risk_percentile_max", "heavy_novelty_min", "cdr3_novelty_min"):
        v = cfg[key]
        if not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"Invalid {key}={v}")
    if not 0 <= cfg["pH"] <= 14 or not 0 <= cfg["risk_percentile_max"] <= 100:
        raise ValueError("pH must be 0..14, percentile 0..100")
    if any(cfg[k] < 0 for k in ("charge_tolerance", "charge_penalty_weight", "heavy_novelty_min", "cdr3_novelty_min")):
        raise ValueError("Tolerance/weights/distances cannot be negative")
    if any(cfg[k] > 1 for k in ("heavy_novelty_min", "cdr3_novelty_min")):
        raise ValueError("Novelty distances must be <= 1")
    if not isinstance(cfg["require_cdr3_in_heavy"], bool):
        raise ValueError("require_cdr3_in_heavy must be boolean")
    metric_columns = {"cdr3_gravy", "cdr3_hydrophobic_run", "cdr3_max_window_gravy_5", "cdr3_max_window_abs_charge_5", "cdr3_len"}
    if set(cfg["upper_tail_metrics"] + cfg["two_sided_metrics"]) != metric_columns or len(cfg["upper_tail_metrics"] + cfg["two_sided_metrics"]) != len(metric_columns):
        raise ValueError("Tail metric lists must partition the five prespecified metrics")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_sequence(value: Any, field: str, row: int) -> str:
    if pd.isna(value):
        raise ValueError(f"Row {row}: missing {field}")
    seq = str(value).strip().upper()
    invalid = sorted(set(seq) - AA)
    if not seq or invalid:
        raise ValueError(f"Row {row}: invalid {field} (empty or noncanonical residues: {invalid})")
    return seq


def validate_table(df: pd.DataFrame, label: str, target: str, require_cdr3: bool) -> pd.DataFrame:
    required = {"Heavy", "cdr3"} | ({"Target"} if label == "reference" else set())
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{label} missing columns {missing}")
    if df.empty:
        raise ValueError(f"{label} is empty")
    out = df.copy().reset_index(drop=True)
    if "Target" not in out:
        out["Target"] = target
    out["Target"] = out["Target"].astype("string").str.strip()
    if out["Target"].isna().any() or (out["Target"] == "").any():
        raise ValueError(f"{label} contains empty Target")
    if label == "candidate" and (out["Target"] != target).any():
        raise ValueError("Candidate Target values must match --target exactly")
    for col in ("Heavy", "cdr3"):
        out[col] = [parse_sequence(v, col, i + 2) for i, v in enumerate(out[col])]
    if require_cdr3:
        bad = out.index[~pd.Series([c in h for h, c in zip(out["Heavy"], out["cdr3"])])].tolist()
        if bad:
            raise ValueError(f"{label}: CDR3 is not a substring of Heavy in CSV rows {[i + 2 for i in bad[:10]]}. Disable this check only for a documented alternate representation.")
    if label == "candidate":
        if "candidate_name" not in out:
            out["candidate_name"] = [f"C{i+1}" for i in range(len(out))]
        if out["candidate_name"].isna().any() or out["candidate_name"].astype(str).str.strip().eq("").any():
            raise ValueError("Candidate names cannot be empty")
        if out["candidate_name"].duplicated().any():
            raise ValueError("Candidate names must be unique")
        if "binding_probability" in out:
            bp = pd.to_numeric(out["binding_probability"], errors="coerce")
            if bp.isna().any() or not np.isfinite(bp.to_numpy()).all() or not bp.between(0, 1).all():
                raise ValueError("binding_probability must be finite and within [0,1]")
            out["binding_probability"] = bp.astype(float)
    return out


def count_overlapping(seq: str, pattern: str) -> int:
    return sum(re.match(pattern, seq[i:]) is not None for i in range(len(seq)))


def longest_run(seq: str, residues: frozenset[str]) -> int:
    best = current = 0
    for a in seq:
        current = current + 1 if a in residues else 0
        best = max(best, current)
    return best


def gravy(seq: str) -> float:
    return float(np.mean([KD[a] for a in seq]))


def charge(seq: str, ph: float) -> float:
    positive = lambda pka: 1 / (1 + 10 ** (ph - pka))
    negative = lambda pka: 1 / (1 + 10 ** (pka - ph))
    total = positive(PKA["Nterm"]) - negative(PKA["Cterm"])
    for aa in seq:
        if aa in "KRH":
            total += positive(PKA[aa])
        elif aa in "DECY":
            total -= negative(PKA[aa])
    return float(total)


def estimate_pi(seq: str) -> float:
    lo, hi = 0.0, 14.0
    for _ in range(55):
        mid = (lo + hi) / 2
        if charge(seq, mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def window_max(seq: str, func, window: int = 5) -> float:
    return max(func(seq[i:i + window]) for i in range(max(1, len(seq) - window + 1))) if len(seq) > window else func(seq)


def fraction(seq: str, residues: frozenset[str]) -> float:
    return sum(x in residues for x in seq) / len(seq)


def compute_features(heavy: str, cdr3: str, ph: float) -> dict[str, float | int]:
    o: dict[str, float | int] = {}
    for prefix, s in (("heavy", heavy), ("cdr3", cdr3)):
        o.update({
            f"{prefix}_len": len(s), f"{prefix}_gravy": gravy(s),
            f"{prefix}_net_charge_pH": charge(s, ph), f"{prefix}_pI": estimate_pi(s),
            f"{prefix}_hydrophobic_run": longest_run(s, HYDROPHOBIC),
            f"{prefix}_frac_hydrophobic": fraction(s, HYDROPHOBIC),
            f"{prefix}_frac_aromatic": fraction(s, AROMATIC),
            f"{prefix}_frac_positive": fraction(s, POS), f"{prefix}_frac_negative": fraction(s, NEG),
            f"{prefix}_n_glyco_motifs": count_overlapping(s, r"N[^P][ST]"),
            f"{prefix}_deamidation_motifs": sum(count_overlapping(s, p) for p in ("NG", "NS")),
            f"{prefix}_isomerization_motifs": sum(count_overlapping(s, p) for p in ("DG", "DS", "DT", "DD")),
            f"{prefix}_methionines": s.count("M"), f"{prefix}_cysteines": s.count("C"),
        })
    o["cdr3_max_window_gravy_5"] = window_max(cdr3, gravy)

    o["cdr3_max_window_abs_charge_5"] = window_max(cdr3, lambda s: abs(charge(s, ph)))
    return o


def with_features(df: pd.DataFrame, ph: float) -> pd.DataFrame:
    features = pd.DataFrame([compute_features(h, c, ph) for h, c in zip(df["Heavy"], df["cdr3"])])
    return pd.concat([df.reset_index(drop=True), features], axis=1)


def hard_filter(row: pd.Series, cfg: dict[str, Any]) -> tuple[bool, str]:
    t = cfg["thresholds"]
    comparisons = [
        (not t["cdr3_len_min"] <= row["cdr3_len"] <= t["cdr3_len_max"], "cdr3_length"),
        (row["cdr3_cysteines"] > t["cdr3_cysteines_max"], "cdr3_cysteines"),
        (row["cdr3_n_glyco_motifs"] > t["cdr3_n_glyco_motifs_max"], "cdr3_glycosylation_sequon"),
        (row["cdr3_deamidation_motifs"] > t["cdr3_deamidation_motifs_max"], "cdr3_deamidation_motif"),
        (row["cdr3_isomerization_motifs"] > t["cdr3_isomerization_motifs_max"], "cdr3_isomerization_motif"),
        (row["cdr3_hydrophobic_run"] > t["cdr3_hydrophobic_run_max"], "cdr3_hydrophobic_run"),
        (row["cdr3_gravy"] > t["cdr3_gravy_max"], "cdr3_gravy"),
        (abs(row["cdr3_net_charge_pH"]) > t["cdr3_abs_charge_max"], "cdr3_charge_proxy"),
        (row["cdr3_methionines"] > t["cdr3_methionines_max"], "cdr3_methionines"),
    ]
    reasons = [name for failed, name in comparisons if failed]
    return not reasons, ";".join(reasons)


def reference_normalization(reference: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, dict[str, float | str]]:
    params: dict[str, dict[str, float | str]] = {}
    for metric in cfg["upper_tail_metrics"] + cfg["two_sided_metrics"]:
        x = reference[metric].to_numpy(dtype=float)
        median = float(np.median(x))
        mad = float(np.median(np.abs(x - median)))
        if mad > 1e-12:
            scale, method = mad / 0.67448975, "MAD_normalized"
        else:
            iqr = float(np.percentile(x, 75) - np.percentile(x, 25))
            std = float(np.std(x))
            if iqr > 1e-12:
                scale, method = iqr / 1.349, "IQR_normalized"
            elif std > 1e-12:
                scale, method = std, "STD"
            else:
               
                scale, method = 1.0, "constant_reference_unit_scale"
        params[metric] = {"median": median, "scale": float(scale), "method": method}
    return params


def risk_score(df: pd.DataFrame, params: dict[str, dict[str, float | str]], cfg: dict[str, Any]) -> np.ndarray:
    scores = np.zeros(len(df), dtype=float)
    for metric, weight in cfg["motif_weights"].items():
        scores += float(weight) * df[metric].to_numpy(dtype=float)
    for metric in cfg["upper_tail_metrics"] + cfg["two_sided_metrics"]:
        param = params[metric]
        z = (df[metric].to_numpy(dtype=float) - float(param["median"])) / float(param["scale"])
        scores += np.abs(z) if metric in cfg["two_sided_metrics"] else np.maximum(z, 0.0)
    scores += cfg["charge_penalty_weight"] * np.maximum(np.abs(df["cdr3_net_charge_pH"].to_numpy(dtype=float)) - cfg["charge_tolerance"], 0)
    return scores


def levenshtein(s: str, t: str) -> int:
    if len(s) < len(t):
        s, t = t, s
    previous = list(range(len(t) + 1))
    for i, a in enumerate(s, 1):
        current = [i]
        for j, b in enumerate(t, 1):
            current.append(min(previous[j] + 1, current[-1] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def nearest_distance(seq: str, refs: list[str]) -> float:

    if not refs:
        raise ValueError("Empty reference list")
    if seq in refs:
        return 0.0
    return float(min(levenshtein(seq, ref) / max(len(seq), len(ref)) for ref in refs))


def score_candidates(reference_raw: pd.DataFrame, candidate_raw: pd.DataFrame, target: str, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    validate_config(cfg)
    ref_all = validate_table(reference_raw, "reference", target, cfg["require_cdr3_in_heavy"])
    ref = ref_all.loc[ref_all["Target"] == target].drop_duplicates(subset=["Heavy", "cdr3"]).reset_index(drop=True)
    if ref.empty:
        raise ValueError(f"No target-matched reference rows for {target!r}")
    candidates = validate_table(candidate_raw, "candidate", target, cfg["require_cdr3_in_heavy"])
    
    ref_features = with_features(ref, cfg["pH"])
    candidate_features = with_features(candidates, cfg["pH"])
    normalization = reference_normalization(ref_features, cfg)
    ref_scores = risk_score(ref_features, normalization, cfg)
    candidate_features["developability_risk_score"] = risk_score(candidate_features, normalization, cfg)
    passes = [hard_filter(row, cfg) for _, row in candidate_features.iterrows()]
    candidate_features["hard_filter_pass"] = [p[0] for p in passes]
    candidate_features["hard_filter_reasons"] = [p[1] for p in passes]
  
    sorted_ref_scores = np.sort(ref_scores)
    candidate_features["developability_risk_score_percentile"] = 100 * np.searchsorted(sorted_ref_scores, candidate_features["developability_risk_score"].to_numpy(dtype=float), side="right") / len(sorted_ref_scores)
    h_refs = list(ref["Heavy"].unique())
    c_refs = list(ref["cdr3"].unique())
    candidate_features["heavy_nn_edit_distance"] = [nearest_distance(s, h_refs) for s in candidate_features["Heavy"]]
    candidate_features["cdr3_nn_edit_distance"] = [nearest_distance(s, c_refs) for s in candidate_features["cdr3"]]
    candidate_features["low_sequence_risk_flag"] = candidate_features["hard_filter_pass"] & (candidate_features["developability_risk_score_percentile"] <= cfg["risk_percentile_max"])
    candidate_features["sequence_novelty_flag"] = (candidate_features["heavy_nn_edit_distance"] >= cfg["heavy_novelty_min"]) | (candidate_features["cdr3_nn_edit_distance"] >= cfg["cdr3_novelty_min"])
    candidate_features["joint_priority_flag"] = candidate_features["low_sequence_risk_flag"] & candidate_features["sequence_novelty_flag"]
   
    sort_cols = ["joint_priority_flag", "low_sequence_risk_flag", "hard_filter_pass"]
    ascending = [False] * len(sort_cols)
    if "binding_probability" in candidate_features:
        sort_cols.append("binding_probability")
        ascending.append(False)
    sort_cols += ["developability_risk_score", "cdr3_nn_edit_distance", "candidate_name"]
    ascending += [True, False, True]
    candidate_features = candidate_features.sort_values(sort_cols, ascending=ascending, kind="stable").reset_index(drop=True)
    candidate_features.insert(0, "rank", np.arange(1, len(candidate_features) + 1))
    meta = {
        "target": target, "reference_n_raw_target": int((ref_all["Target"] == target).sum()),
        "reference_n_unique": len(ref), "candidate_n": len(candidates),
        "reference_score_summary": {"min": float(ref_scores.min()), "median": float(np.median(ref_scores)), "max": float(ref_scores.max())},
        "reference_normalization": normalization,
        "counts": {name: int(candidate_features[name].sum()) for name in ("hard_filter_pass", "low_sequence_risk_flag", "sequence_novelty_flag", "joint_priority_flag")},
        "sorting": {"columns": sort_cols, "ascending": ascending},
        "interpretation": "Heuristic reference-relative sequence ranking only; flags are NOT experimental developability or efficacy claims.",
    }
    return candidate_features, meta


def cli() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--reference_csv", type=Path)
    p.add_argument("--candidate_csv", type=Path)
    p.add_argument("--target")
    p.add_argument("--output_csv", type=Path, default=Path("developability_ranked.csv"))
    p.add_argument("--config_json", type=Path, help="Full JSON config. Generate the default with --write_default_config.")
    p.add_argument("--write_default_config", type=Path, help="Write default config JSON and exit")
    p.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING"], default="INFO")
    args = p.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")
    if args.write_default_config:
        args.write_default_config.parent.mkdir(parents=True, exist_ok=True)
        args.write_default_config.write_text(json.dumps(DEFAULT_CONFIG, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.write_default_config}")
        return
    if not all((args.reference_csv, args.candidate_csv, args.target)):
        p.error("--reference_csv, --candidate_csv and --target are required for scoring")
    cfg = json.loads(args.config_json.read_text(encoding="utf-8")) if args.config_json else DEFAULT_CONFIG
    validate_config(cfg)
    for path in (args.reference_csv, args.candidate_csv):
        if not path.is_file():
            p.error(f"Input not found: {path}")
    ranked, meta = score_candidates(pd.read_csv(args.reference_csv), pd.read_csv(args.candidate_csv), args.target, cfg)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(args.output_csv, index=False, float_format="%.10g")
    meta.update({
        "utc_created": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pd.__version__, "config": cfg,
        "reference_file_sha256": sha256(args.reference_csv),
        "candidate_file_sha256": sha256(args.candidate_csv),
    })
    metadata_path = args.output_csv.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    LOG.info("Target=%s reference_unique=%d candidates=%d", args.target, meta["reference_n_unique"], meta["candidate_n"])
    LOG.info("Hard-filter pass=%d joint priority=%d", meta["counts"]["hard_filter_pass"], meta["counts"]["joint_priority_flag"])
    LOG.info("Wrote %s and %s", args.output_csv, metadata_path)


if __name__ == "__main__":
    cli()
