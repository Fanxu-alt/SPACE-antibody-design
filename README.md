# SPACE: A Unified Framework for Multi-Constraint Antigen-Specific Antibody Design Operating in Sequence Space

A sequence-driven closed-loop framework for antigen-specific antibody design that integrates:
- Antigen-conditioned CDRH3 generation
- Sequence-based antibody–antigen interaction prediction
- Developability-aware candidate prioritization
- Interactive Gradio Web Application
  
<p align="center">
  <img src="data/raw/fig1.png" width="700">
</p>

## Framework Architecture
Antigen sequence  
→ Conditional VAE (CDRH3 generation)  
→ Generated CDRH3 candidates  
→ ESM-2 Cross-Attention Model (binding prediction)  
→ Developability-aware ranking  
→ Final antibody candidates

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
- torch
- Transformers ≥ 4.30
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- gradio
- gradio_client
- ANARCI (for CDRH3 extraction)

## Quick Start

### Train models

```bash
python code/train/train_cdrh3_vae.py
python code/train/train_conditional_cvae.py
python code/train/train_esm2_cross_attention.py
```
## Web Application

We extend the framework into a goal-oriented antibody design agent.

Given:
- an antigen sequence,
- a heavy-chain template,
- a CDRH3 template,
- and user-defined design constraints,

the agent automatically:
- generates candidate CDRH3 sequences,
- predicts antibody–antigen binding,
- filters and ranks candidates using developability criteria,
- and iterates until the design requirements are met or the maximum number of rounds is reached.

### Launch locally

```bash
python Antibody_Design_Application/app_gradio.py
```

Open:

```bash
http://127.0.0.1:7860
```

## Online demo

A public online demo is available at:

https://huggingface.co/spaces/Fanxu-alt/Antibody-Design-App

### Notes
The online demo is provided for accessibility and interactive exploration. Availability may depend on deployment resources and API configuration.

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

## Dataset: covid_human_heavy_cdr3_aa_unique_len4_30.txt

This file contains a non-redundant collection of human SARS-CoV-2-associated heavy-chain CDRH3 amino acid sequences curated from the Observed Antibody Space (OAS) database.

### Description
- **Processing steps**: 
  1. removed empty entries
  2. removed sequences containing non-canonical amino acid characters
  3. removed duplicate sequences globally across all files
  4. retained only sequences with lengths between **4 and 30 amino acids**
     
Download: https://drive.google.com/file/d/1n46ld31QrC9oYlZVsR7JZsoOgX_TFupc/view?usp=drive_link.

### Contact

If you have any questions about this repository, please contact:

**[Fanxu Meng](mailto:f.meng@vu.nl)**
