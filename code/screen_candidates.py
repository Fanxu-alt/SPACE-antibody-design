#!/usr/bin/env python3
"""
Developability screening and ranking for SPACE antibody candidates.

Input candidate CSV should contain:
- Target
- Heavy
- cdr3

Optional:
- candidate_name
- binding_probability
- binding_logit

"""

import argparse
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

KD_SCALE = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, "A": 1.8,
    "G": -0.4, "T": -0.7, "S": -0.8, "W": -0.9, "Y": -1.3, "P": -1.6,
    "H": -3.2, "E": -3.5, "Q": -3.5, "D": -3.5, "N": -3.5, "K": -3.9,
    "R": -4.5,
}

PKA = {
    "Cterm": 3.1,
    "Nterm": 8.0,
    "C": 8.5,
    "D": 3.9,
    "E": 4.1,
    "H": 6.0,
    "K": 10.5,
    "R": 12.5,
    "Y": 10.1,
}

POSITIVE = set("KRH")
NEGATIVE = set("DE")
HYDROPHOBIC = set("AILMFWVYC")
AROMATIC = set("FWY")


ABS_THRESHOLDS = {
    "cdr3_len_min": 8,
    "cdr3_len_max": 20,
    "extra_cys_heavy_max": 0,
    "extra_cys_cdr3_max": 0,
    "n_glyco_motif_heavy_max": 0,
    "n_glyco_motif_cdr3_max": 0,
    "deamidation_motif_cdr3_max": 1,
    "isomerization_motif_cdr3_max": 1,
    "max_hydrophobic_run_cdr3_max": 5,
    "gravy_cdr3_max": 0.8,
    "net_charge_cdr3_abs_max": 4,
    "oxidation_m_count_cdr3_max": 1,
}


def clean_seq(seq: str) -> str:
    seq = str(seq or "").strip().upper()
    return "".join(ch for ch in seq if ch in AMINO_ACIDS)


def clean_target(x: str) -> str:
    return str(x or "").strip()


def count_motif_regex(seq: str, pattern: str) -> int:
    return len(list(re.finditer(pattern, seq)))


def count_deamidation_motifs(seq: str) -> int:
    return seq.count("NG") + seq.count("NS")


def count_isomerization_motifs(seq: str) -> int:
    return sum(seq.count(m) for m in ("DG", "DS", "DT", "DD"))


def count_oxidation_hotspots(seq: str) -> int:
    return seq.count("M")


def count_extra_cys(seq: str, expected_min: int = 2) -> int:
    return max(0, seq.count("C") - expected_min)


def hydrophobic_run_length(seq: str) -> int:
    longest = 0
    current = 0

    for aa in seq:
        if aa in HYDROPHOBIC:
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    return longest


def gravy(seq: str) -> float:
    if not seq:
        return np.nan
    return float(np.mean([KD_SCALE[a] for a in seq]))


def net_charge_at_ph(seq: str, ph: float = 7.4) -> float:
    if not seq:
        return np.nan

    def pos_fraction(pka_val: float) -> float:
        return 1.0 / (1.0 + 10 ** (ph - pka_val))

    def neg_fraction(pka_val: float) -> float:
        return 1.0 / (1.0 + 10 ** (pka_val - ph))

    charge = 0.0
    charge += pos_fraction(PKA["Nterm"])
    charge -= neg_fraction(PKA["Cterm"])

    for aa in seq:
        if aa == "K":
            charge += pos_fraction(PKA["K"])
        elif aa == "R":
            charge += pos_fraction(PKA["R"])
        elif aa == "H":
            charge += pos_fraction(PKA["H"])
        elif aa == "D":
            charge -= neg_fraction(PKA["D"])
        elif aa == "E":
            charge -= neg_fraction(PKA["E"])
        elif aa == "C":
            charge -= neg_fraction(PKA["C"])
        elif aa == "Y":
            charge -= neg_fraction(PKA["Y"])

    return float(charge)


def estimate_pI(seq: str) -> float:
    lo, hi = 0.0, 14.0

    for _ in range(60):
        mid = 0.5 * (lo + hi)
        charge = net_charge_at_ph(seq, ph=mid)
        if charge > 0:
            lo = mid
        else:
            hi = mid

    return float(0.5 * (lo + hi))


def rolling_window_max(seq: str, func, window: int = 5) -> float:
    if len(seq) == 0:
        return np.nan

    if len(seq) <= window:
        return func(seq)

    values = [func(seq[i:i + window]) for i in range(len(seq) - window + 1)]
    return float(np.max(values))


def robust_zscore(series: pd.Series) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce")
    med = series.median()
    mad = np.median(np.abs(series - med))

    if mad < 1e-12 or np.isnan(mad):
        return pd.Series(np.zeros(len(series)), index=series.index)

    return 0.6745 * (series - med) / mad


def levenshtein(s1: str, s2: str) -> int:
    if s1 == s2:
        return 0

    if len(s1) < len(s2):
        s1, s2 = s2, s1

    previous = list(range(len(s2) + 1))

    for i, c1 in enumerate(s1, start=1):
        current = [i]

        for j, c2 in enumerate(s2, start=1):
            insertion = previous[j] + 1
            deletion = current[j - 1] + 1
            substitution = previous[j - 1] + int(c1 != c2)
            current.append(min(insertion, deletion, substitution))

        previous = current

    return previous[-1]


def normalized_edit_distance(s1: str, s2: str) -> float:
    denom = max(len(s1), len(s2), 1)
    return levenshtein(s1, s2) / denom


def nearest_neighbor_distance(
    query_seq: str,
    ref_seqs: List[str],
    exclude_identical: bool = True,
) -> float:
    best = None

    for ref in ref_seqs:
        if exclude_identical and query_seq == ref:
            continue

        dist = normalized_edit_distance(query_seq, ref)

        if best is None or dist < best:
            best = dist

    return np.nan if best is None else float(best)


@dataclass
class DevFeatures:
    heavy_len: int
    cdr3_len: int
    heavy_gravy: float
    cdr3_gravy: float
    heavy_net_charge_pH74: float
    cdr3_net_charge_pH74: float
    heavy_pI: float
    cdr3_pI: float
    heavy_hydrophobic_run: int
    cdr3_hydrophobic_run: int
    heavy_frac_hydrophobic: float
    cdr3_frac_hydrophobic: float
    heavy_frac_aromatic: float
    cdr3_frac_aromatic: float
    heavy_frac_positive: float
    cdr3_frac_positive: float
    heavy_frac_negative: float
    cdr3_frac_negative: float
    heavy_n_glyco_motifs: int
    cdr3_n_glyco_motifs: int
    heavy_deamidation_motifs: int
    cdr3_deamidation_motifs: int
    heavy_isomerization_motifs: int
    cdr3_isomerization_motifs: int
    heavy_oxidation_m_count: int
    cdr3_oxidation_m_count: int
    heavy_extra_cys_proxy: int
    cdr3_extra_cys_proxy: int
    cdr3_max_window_gravy_5: float
    cdr3_max_window_abs_charge_5: float


def compute_features(heavy: str, cdr3: str) -> DevFeatures:
    heavy = clean_seq(heavy)
    cdr3 = clean_seq(cdr3)

    def frac(seq: str, aa_set: set) -> float:
        return sum(aa in aa_set for aa in seq) / max(len(seq), 1)

    def abs_charge_local(seq: str) -> float:
        return abs(net_charge_at_ph(seq, ph=7.4))

    return DevFeatures(
        heavy_len=len(heavy),
        cdr3_len=len(cdr3),
        heavy_gravy=gravy(heavy),
        cdr3_gravy=gravy(cdr3),
        heavy_net_charge_pH74=net_charge_at_ph(heavy, 7.4),
        cdr3_net_charge_pH74=net_charge_at_ph(cdr3, 7.4),
        heavy_pI=estimate_pI(heavy),
        cdr3_pI=estimate_pI(cdr3),
        heavy_hydrophobic_run=hydrophobic_run_length(heavy),
        cdr3_hydrophobic_run=hydrophobic_run_length(cdr3),
        heavy_frac_hydrophobic=frac(heavy, HYDROPHOBIC),
        cdr3_frac_hydrophobic=frac(cdr3, HYDROPHOBIC),
        heavy_frac_aromatic=frac(heavy, AROMATIC),
        cdr3_frac_aromatic=frac(cdr3, AROMATIC),
        heavy_frac_positive=frac(heavy, POSITIVE),
        cdr3_frac_positive=frac(cdr3, POSITIVE),
        heavy_frac_negative=frac(heavy, NEGATIVE),
        cdr3_frac_negative=frac(cdr3, NEGATIVE),
        heavy_n_glyco_motifs=count_motif_regex(heavy, r"N[^P][ST]"),
        cdr3_n_glyco_motifs=count_motif_regex(cdr3, r"N[^P][ST]"),
        heavy_deamidation_motifs=count_deamidation_motifs(heavy),
        cdr3_deamidation_motifs=count_deamidation_motifs(cdr3),
        heavy_isomerization_motifs=count_isomerization_motifs(heavy),
        cdr3_isomerization_motifs=count_isomerization_motifs(cdr3),
        heavy_oxidation_m_count=count_oxidation_hotspots(heavy),
        cdr3_oxidation_m_count=count_oxidation_hotspots(cdr3),
        heavy_extra_cys_proxy=count_extra_cys(heavy, expected_min=2),
        cdr3_extra_cys_proxy=max(0, cdr3.count("C")),
        cdr3_max_window_gravy_5=rolling_window_max(cdr3, gravy, window=5),
        cdr3_max_window_abs_charge_5=rolling_window_max(cdr3, abs_charge_local, window=5),
    )


def features_to_dict(features: DevFeatures) -> Dict:
    return asdict(features)


def hard_filter_rule_row(row: pd.Series) -> Tuple[bool, List[str]]:
    reasons = []

    if not (ABS_THRESHOLDS["cdr3_len_min"] <= row["cdr3_len"] <= ABS_THRESHOLDS["cdr3_len_max"]):
        reasons.append("CDRH3 length outside allowed range")

    if row["heavy_extra_cys_proxy"] > ABS_THRESHOLDS["extra_cys_heavy_max"]:
        reasons.append("extra cysteine(s) in heavy chain")

    if row["cdr3_extra_cys_proxy"] > ABS_THRESHOLDS["extra_cys_cdr3_max"]:
        reasons.append("cysteine present in CDRH3")

    if row["heavy_n_glyco_motifs"] > ABS_THRESHOLDS["n_glyco_motif_heavy_max"]:
        reasons.append("N-linked glycosylation motif in heavy chain")

    if row["cdr3_n_glyco_motifs"] > ABS_THRESHOLDS["n_glyco_motif_cdr3_max"]:
        reasons.append("N-linked glycosylation motif in CDRH3")

    if row["cdr3_deamidation_motifs"] > ABS_THRESHOLDS["deamidation_motif_cdr3_max"]:
        reasons.append("excess deamidation motif(s) in CDRH3")

    if row["cdr3_isomerization_motifs"] > ABS_THRESHOLDS["isomerization_motif_cdr3_max"]:
        reasons.append("excess Asp isomerization motif(s) in CDRH3")

    if row["cdr3_hydrophobic_run"] > ABS_THRESHOLDS["max_hydrophobic_run_cdr3_max"]:
        reasons.append("long hydrophobic run in CDRH3")

    if row["cdr3_gravy"] > ABS_THRESHOLDS["gravy_cdr3_max"]:
        reasons.append("high average hydrophobicity in CDRH3")

    if abs(row["cdr3_net_charge_pH74"]) > ABS_THRESHOLDS["net_charge_cdr3_abs_max"]:
        reasons.append("extreme net charge in CDRH3")

    if row["cdr3_oxidation_m_count"] > ABS_THRESHOLDS["oxidation_m_count_cdr3_max"]:
        reasons.append("too many methionine oxidation hotspots in CDRH3")

    return len(reasons) == 0, reasons


def relative_risk_score(df: pd.DataFrame) -> pd.Series:
    score = pd.Series(np.zeros(len(df)), index=df.index, dtype=float)

    score += 3.0 * df["heavy_n_glyco_motifs"]
    score += 4.0 * df["cdr3_n_glyco_motifs"]
    score += 2.0 * df["cdr3_deamidation_motifs"]
    score += 2.0 * df["cdr3_isomerization_motifs"]
    score += 3.0 * df["heavy_extra_cys_proxy"]
    score += 4.0 * df["cdr3_extra_cys_proxy"]
    score += 0.75 * df["cdr3_oxidation_m_count"]

    outlier_metrics = [
        "cdr3_gravy",
        "cdr3_hydrophobic_run",
        "cdr3_max_window_gravy_5",
        "cdr3_max_window_abs_charge_5",
        "cdr3_len",
        "heavy_n_glyco_motifs",
        "cdr3_n_glyco_motifs",
        "cdr3_deamidation_motifs",
        "cdr3_isomerization_motifs",
        "heavy_extra_cys_proxy",
        "cdr3_extra_cys_proxy",
    ]

    for metric in outlier_metrics:
        rz = robust_zscore(df[metric])
        score += np.maximum(rz, 0.0).astype(float)

    charge_penalty = np.maximum(np.abs(df["cdr3_net_charge_pH74"]) - 2.5, 0.0)
    score += 0.75 * charge_penalty

    return score


class DevelopabilityRanker:
    def __init__(self, reference_csv: str):
        self.reference_csv = reference_csv
        self.df = pd.read_csv(reference_csv)

        required = ["Target", "Heavy", "cdr3"]
        missing = [c for c in required if c not in self.df.columns]

        if missing:
            raise ValueError(f"Missing required columns in {reference_csv}: {missing}")

        self.df = self.df[required].dropna().copy()
        self.df["Target"] = self.df["Target"].map(clean_target)
        self.df["Heavy"] = self.df["Heavy"].map(clean_seq)
        self.df["cdr3"] = self.df["cdr3"].map(clean_seq)

        self.df = self.df[
            (self.df["Target"].str.len() > 0)
            & (self.df["Heavy"].str.len() > 0)
            & (self.df["cdr3"].str.len() > 0)
        ].drop_duplicates(subset=["Target", "Heavy", "cdr3"]).reset_index(drop=True)

    def list_targets(self) -> List[str]:
        return sorted(self.df["Target"].dropna().unique().tolist())

    def get_target_cohort(self, target_name: str) -> pd.DataFrame:
        target_name = clean_target(target_name)
        cohort = self.df[self.df["Target"] == target_name].copy()

        if len(cohort) == 0:
            available = ", ".join(self.list_targets()[:20])
            raise ValueError(
                f"No reference cohort found for Target='{target_name}'. "
                f"Available targets include: {available}"
            )

        return cohort.reset_index(drop=True)

    def attach_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features = [
            features_to_dict(compute_features(h, c))
            for h, c in zip(df["Heavy"], df["cdr3"])
        ]

        feature_df = pd.DataFrame(features)

        out = pd.concat(
            [df.reset_index(drop=True), feature_df.reset_index(drop=True)],
            axis=1,
        )

        pass_values = []
        reason_values = []

        for _, row in out.iterrows():
            passed, reasons = hard_filter_rule_row(row)
            pass_values.append(passed)
            reason_values.append("; ".join(reasons) if reasons else "")

        out["hard_filter_pass"] = pass_values
        out["hard_filter_reasons"] = reason_values

        return out

    def prepare_candidates(self, candidate_df: pd.DataFrame, target_name: str) -> pd.DataFrame:
        df = candidate_df.copy()

        if "Target" not in df.columns:
            df["Target"] = target_name

        if "candidate_name" not in df.columns:
            df["candidate_name"] = [f"C{i + 1}" for i in range(len(df))]

        required = ["Target", "Heavy", "cdr3", "candidate_name"]
        missing = [c for c in required if c not in df.columns]

        if missing:
            raise ValueError(f"Missing required candidate columns: {missing}")

        df = df[required + [c for c in df.columns if c not in required]].copy()
        df["Target"] = df["Target"].map(clean_target)
        df["Heavy"] = df["Heavy"].map(clean_seq)
        df["cdr3"] = df["cdr3"].map(clean_seq)

        df = df[
            (df["Heavy"].str.len() > 0)
            & (df["cdr3"].str.len() > 0)
        ].reset_index(drop=True)

        return self.attach_features(df)

    def score_candidates(self, target_name: str, candidate_df: pd.DataFrame) -> pd.DataFrame:
        cohort_df = self.attach_features(self.get_target_cohort(target_name))
        selected_df = self.prepare_candidates(candidate_df, target_name)

        combined = pd.concat(
            [
                cohort_df.assign(_source="cohort"),
                selected_df.assign(_source="selected"),
            ],
            ignore_index=True,
            sort=False,
        )

        combined["developability_risk_score"] = relative_risk_score(combined)

        selected_scored = combined[combined["_source"] == "selected"].drop(
            columns=["_source"]
        ).reset_index(drop=True)

        cohort_scored = combined[combined["_source"] == "cohort"].copy()
        risk_ref = cohort_scored["developability_risk_score"].dropna().values

        heavy_refs = cohort_df["Heavy"].tolist()
        cdr3_refs = cohort_df["cdr3"].tolist()

        selected_scored["heavy_nn_edit_distance"] = selected_scored["Heavy"].map(
            lambda x: nearest_neighbor_distance(x, heavy_refs, exclude_identical=True)
        )

        selected_scored["cdr3_nn_edit_distance"] = selected_scored["cdr3"].map(
            lambda x: nearest_neighbor_distance(x, cdr3_refs, exclude_identical=True)
        )

        selected_scored["developability_risk_score_percentile"] = selected_scored[
            "developability_risk_score"
        ].map(lambda x: 100.0 * (risk_ref <= x).mean() if len(risk_ref) else np.nan)

        selected_scored["low_risk_claim"] = (
            selected_scored["hard_filter_pass"]
            & (selected_scored["developability_risk_score_percentile"] <= 50.0)
        )

        selected_scored["high_diversity_claim"] = (
            (selected_scored["heavy_nn_edit_distance"] >= 0.10)
            | (selected_scored["cdr3_nn_edit_distance"] >= 0.20)
        )

        selected_scored["overall_claim"] = (
            selected_scored["low_risk_claim"]
            & selected_scored["high_diversity_claim"]
        )

        sort_cols = ["overall_claim", "low_risk_claim", "hard_filter_pass"]

        if "binding_probability" in selected_scored.columns:
            sort_cols.append("binding_probability")

        sort_cols += ["developability_risk_score", "cdr3_nn_edit_distance"]

        ascending = [False, False, False]

        if "binding_probability" in selected_scored.columns:
            ascending.append(False)

        ascending += [True, False]

        selected_scored = selected_scored.sort_values(
            by=sort_cols,
            ascending=ascending,
        ).reset_index(drop=True)

        selected_scored.insert(0, "rank", range(1, len(selected_scored) + 1))

        return selected_scored


def main():
    parser = argparse.ArgumentParser(
        description="Run developability screening and ranking for antibody candidates."
    )

    parser.add_argument(
        "--reference_csv",
        required=True,
        help="Reference antibody CSV containing Target, Heavy and cdr3 columns.",
    )

    parser.add_argument(
        "--candidate_csv",
        required=True,
        help="Candidate antibody CSV containing Heavy and cdr3 columns.",
    )

    parser.add_argument(
        "--target",
        required=True,
        help="Target name used to select the same-antigen reference cohort.",
    )

    parser.add_argument(
        "--output_csv",
        default="developability_scores.csv",
        help="Output CSV file.",
    )

    args = parser.parse_args()

    ranker = DevelopabilityRanker(args.reference_csv)
    candidates = pd.read_csv(args.candidate_csv)

    scored = ranker.score_candidates(
        target_name=args.target,
        candidate_df=candidates,
    )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_path, index=False)

    print(f"Saved developability scores to: {output_path}")
    print(scored.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
