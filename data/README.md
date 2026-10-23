# Data Directory

## Structure

```
data/
├── raw/
│   └── hc3_raw.parquet          ← Original HC3 data (auto-downloaded)
└── processed/
    ├── hc3_flat.parquet         ← Flat one-row-per-answer format
    ├── hc3_metadata.json        ← Dataset statistics and provenance
    ├── train.parquet            ← Training split (70%)
    ├── val.parquet              ← Validation split (15%)
    └── test.parquet             ← Test split (15%)
```

## Dataset: HC3 — Human ChatGPT Comparison Corpus

| Item | Detail |
|---|---|
| **Full name** | Human ChatGPT Comparison Corpus (HC3) |
| **HuggingFace ID** | `Hello-SimpleAI/HC3` |
| **Paper** | Guo et al. (2023), arXiv:2301.07597 |
| **License** | CC BY-SA 4.0 |
| **Download URL** | https://huggingface.co/datasets/Hello-SimpleAI/HC3 |

## Data Generation Command

```bash
python -m src.data_loader
```

This will:
1. Download HC3 from HuggingFace
2. Flatten nested answer lists (one row per answer)
3. Remove empty and duplicate texts
4. Add text-length columns
5. Split into train/val/test (70/15/15)
6. Save all files to `data/processed/`

## ⚠️ Note

The `test.parquet` file must NOT be used during model training or
hyperparameter tuning. It is reserved exclusively for final evaluation.
