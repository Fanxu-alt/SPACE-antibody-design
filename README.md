# SPACE-antibody-design

Implementation of

SPACE: A Unified Framework for Multi-Constraint Antigen-Specific Antibody Design Operating in Sequence Space

SPACE consists of:

- H3-AbSeqVAE (antigen-conditioned CDRH3 generation)
- AbAgBinder (antibody-antigen interaction prediction)
- Developability-aware candidate prioritization
  
<p align="center">
  <img src="data/raw/fig1.png" width="700">
</p>

## Framework Architecture

1. VAE Pretraining
2. Conditional VAE Fine-tuning
3. Binding Model Training
4. Antibody Generation
5. Binding Prediction
6. Developability Assessment

## Quick Start

### Train models

```bash
python code/train/train_cdrh3_vae.py
python code/train/train_conditional_cvae.py
python code/train/train_esm2_cross_attention.py
```

### Online Web Application

website: https://antibody-design.vercel.app

The backend is implemented using FastAPI and deployed on Hugging Face Spaces.

## Pretrained Models

The checkpoint provided in this repository:
```bash
checkpoints/best_esm2_cross_attention.pt
```
was trained using the small ESM2 model (esm2_t12_35M_UR50D) for demonstration and reproducibility.

Due to file size limitations, checkpoints trained with larger ESM2 models (esm2_t33_650M_UR50D) are hosted on Google Drive:

Download larger pretrained models (best_esm2_cross_attention_regression_fixed_antigen.pt, best_esm2_cross_attention_regression.pt, best_esm2_cross_attention.pt): 

https://drive.google.com/file/d/14ZK1tzs6QaPVj8i74B2Rzhb3JpxOE25r/view?usp=drive_link, https://drive.google.com/file/d/1ZZQzJYHQ37Zc1KjwqAsiiYMyB8yyORGY/view?usp=drive_link, https://drive.google.com/file/d/1SdkpORkcsUErk5c2iiNBYlkyTVKrbPLN/view?usp=drive_link.

After downloading, place the checkpoints in the `checkpoints/` directory.

## Hardware Requirements
### Recommended
- GPU: NVIDIA A100
- CPU: ≥8 cores
- RAM: ≥32 GB
- Storage: ≥20 GB

### System Requirements
- OS: Ubuntu 20.04 / Linux / macOS
- Python ≥ 3.9

### Dependencies
- fastapi
- uvicorn
- pydantic
- pandas
- numpy
- torch
- Transformers ≥ 4.30
- scikit-learn
- matplotlib
- seaborn
- openai
- ANARCI (for CDRH3 extraction)

## Dataset: covid_human_heavy_cdr3_aa_unique_len4_30.txt

This file contains a non-redundant collection of human SARS-CoV-2-associated heavy-chain CDRH3 amino acid sequences curated from the Observed Antibody Space (OAS) database.

### Description
- **Processing steps**: 
  1. removed empty entries
  2. removed sequences containing non-canonical amino acid characters
  3. removed duplicate sequences globally across all files
  4. retained only sequences with lengths between **4 and 30 amino acids**
     
Download: https://drive.google.com/file/d/1n46ld31QrC9oYlZVsR7JZsoOgX_TFupc/view?usp=drive_link.

## Dataset Construction
We also collected antibody–antigen complex structures from the SAbDab database: HIV gp120, Influenza Hemagglutinin (HA), HIV gp160, Plasmodium Circumsporozoite Protein (CSP), Influenza Neuraminidase (NA).


All antigen-specific datasets are available under:

```bash
data/raw/
```

For each complex, the IMGT-numbered CDRH3 region were extracted. Negative samples were generated using a dissimilarity-based negative sampling strategy (sequence identity threshold of less than 60%).

### Contact

If you have any questions about this repository, please contact:

**[Fanxu Meng](mailto:f.meng@vu.nl)**
