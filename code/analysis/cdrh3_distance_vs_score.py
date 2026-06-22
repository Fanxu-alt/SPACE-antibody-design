import re
import pandas as pd
import numpy as np
import Levenshtein
from scipy.stats import spearmanr, pearsonr

GENERATED_FILE = "generated_cdrh3_from_antigenfinetune.txt"
RANKED_FILE = "ranked_cdrh3_candidates.csv"
NATURAL_FILE = "CoV-AbDab_only_sars2_filter_only_1_cdr3.csv"

TARGET_ANTIGEN = (
"RVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNF"
)

OUTPUT_FILE = "cdrh3_distance_vs_score.csv"

def read_generated_cdrh3(path):

    seqs = []
    pattern = re.compile(r"^\d+\s+len=\d+\s+([A-Z]+)")

    with open(path) as f:
        for line in f:
            line = line.strip()
            m = pattern.match(line)
            if m:
                seqs.append(m.group(1))

    return list(set(seqs))

def read_natural_cdrh3(csv_file):

    df = pd.read_csv(csv_file)

    df = df[df["antigen"] == TARGET_ANTIGEN]

    natural = df["cdr3"].dropna().unique().tolist()

    return natural

def min_distance(seq, natural_list):

    distances = [Levenshtein.distance(seq, nat) for nat in natural_list]

    return min(distances)

def main():

    print("Loading generated CDRH3...")
    generated = read_generated_cdrh3(GENERATED_FILE)
    print("Generated:", len(generated))


    print("Loading natural CDRH3...")
    natural = read_natural_cdrh3(NATURAL_FILE)
    print("Natural:", len(natural))


    print("Loading scores...")
    ranked = pd.read_csv(RANKED_FILE)

    results = []

    print("Computing distances...")

    for seq in generated:

        d = min_distance(seq, natural)

        score = ranked.loc[ranked["cdrh3"] == seq, "binding_score"]

        if len(score) == 0:
            continue

        score = float(score.values[0])

        results.append({
            "cdrh3": seq,
            "min_levenshtein_distance": d,
            "binding_score": score
        })


    df = pd.DataFrame(results)

    df.to_csv(OUTPUT_FILE, index=False)

    print("\nSaved:", OUTPUT_FILE)

    pearson = pearsonr(df["min_levenshtein_distance"], df["binding_score"])
    spearman = spearmanr(df["min_levenshtein_distance"], df["binding_score"])

    print("\nCorrelation:")

    print("Pearson:", pearson)
    print("Spearman:", spearman)


if __name__ == "__main__":
    main()
