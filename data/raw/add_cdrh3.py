import json
import pandas as pd
from anarcii import Anarcii

input_file = "sabdab_positive_heavy_antigen.csv"
output_file = "sabdab_positive_heavy_antigen_cdrh3_imgt.csv"

model = Anarcii()
df = pd.read_csv(input_file)

def extract_cdrh3_imgt(seq):
    try:
        result = model.number([seq])
        item = result["Sequence 1"]

        if item.get("error") is not None:
            return None, None, None

        if item.get("chain_type") != "H":
            return None, None, None

        cdr = []
        numbering = []

        for (pos, ins), aa in item["numbering"]:
            if 105 <= pos <= 117 and aa != "-":
                pos_label = str(pos) + (ins.strip() if ins.strip() else "")
                cdr.append(aa)
                numbering.append([pos_label, aa])

        if not cdr:
            return None, None, None

        return "".join(cdr), json.dumps(numbering), len(cdr)

    except Exception:
        return None, None, None

cdrh3_seqs = []
cdrh3_numberings = []
cdrh3_lens = []

for i, seq in enumerate(df["heavy_seq"]):
    cdrh3, numbering, length = extract_cdrh3_imgt(seq)
    cdrh3_seqs.append(cdrh3)
    cdrh3_numberings.append(numbering)
    cdrh3_lens.append(length)

    if (i + 1) % 500 == 0:
        print("processed", i + 1)

df["cdrh3_seq"] = cdrh3_seqs
df["cdrh3_imgt_numbering"] = cdrh3_numberings
df["cdrh3_len"] = cdrh3_lens

df.to_csv(output_file, index=False)

print("原始样本:", len(df))
print("成功提取 CDRH3:", df["cdrh3_seq"].notna().sum())
print("失败:", df["cdrh3_seq"].isna().sum())
print("保存:", output_file)

print(df[["pdb", "antigen_group", "cdrh3_seq", "cdrh3_len"]].head())
