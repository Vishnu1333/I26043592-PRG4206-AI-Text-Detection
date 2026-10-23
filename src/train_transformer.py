"""
src/train_transformer.py — DistilBERT fine-tuning for AI text detection.
PRG4206 | AI-Generated Text Detection System

Model 4: DistilBERT sequence classifier
  - Fine-tuned on HC3 training split
  - Supports CPU and Apple Silicon MPS
  - Configurable for low-resource / student hardware
"""

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import (
    BERT_DIR,
    DISTILBERT_BATCH_SIZE,
    DISTILBERT_EPOCHS,
    DISTILBERT_LR,
    DISTILBERT_MAX_LENGTH,
    DISTILBERT_MODEL_NAME,
    DISTILBERT_WEIGHT_DECAY,
    RANDOM_STATE,
    ensure_dirs,
)
from src.evaluate import (
    compute_metrics,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
    print_classification_report,
    save_metrics,
    save_predictions,
)
from src.utils import get_logger, set_seed

logger = get_logger(__name__)


def _get_device():
    """Detect the best available compute device."""
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("Device: CUDA GPU")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Device: Apple Silicon MPS")
        else:
            device = torch.device("cpu")
            logger.info("Device: CPU (no GPU detected)")
        return device
    except ImportError:
        raise RuntimeError("PyTorch is not installed. Run: pip install torch")


class HC3Dataset:
    """PyTorch Dataset for HC3 text classification."""

    def __init__(self, texts, labels, tokenizer, max_length: int):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def train_distilbert(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
    text_col:  str = "clean_text",
    label_col: str = "label",
    epochs:     int   = DISTILBERT_EPOCHS,
    batch_size: int   = DISTILBERT_BATCH_SIZE,
    lr:         float = DISTILBERT_LR,
    max_length: int   = DISTILBERT_MAX_LENGTH,
) -> dict:
    """Fine-tune DistilBERT for binary (human / AI) classification.

    Args:
        train_df:   Training DataFrame.
        val_df:     Validation DataFrame.
        test_df:    Test DataFrame.
        text_col:   Column containing (cleaned) text.
        label_col:  Column containing binary labels (0 / 1).
        epochs:     Number of fine-tuning epochs.
        batch_size: Per-device batch size.
        lr:         Learning rate.
        max_length: Maximum token sequence length.

    Returns:
        Dictionary with model_dir, val_metrics, test_metrics.
    """
    try:
        import torch
        from torch.utils.data import DataLoader
        from torch.optim import AdamW
        from transformers import (
            DistilBertTokenizerFast,
            DistilBertForSequenceClassification,
            get_linear_schedule_with_warmup,
        )
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch and HuggingFace Transformers are required. "
            "Run: pip install torch transformers"
        ) from exc

    set_seed(RANDOM_STATE)
    ensure_dirs()
    logger.info("=== Training DistilBERT ===")
    logger.info("Epochs: %d | Batch: %d | LR: %g | Max length: %d",
                epochs, batch_size, lr, max_length)

    device = _get_device()

    # ── Tokeniser ─────────────────────────────────────────────
    logger.info("Loading tokeniser: %s …", DISTILBERT_MODEL_NAME)
    tokenizer = DistilBertTokenizerFast.from_pretrained(DISTILBERT_MODEL_NAME)

    # ── Datasets ──────────────────────────────────────────────
    def make_loader(df: pd.DataFrame, shuffle: bool) -> DataLoader:
        texts  = df[text_col].tolist()
        labels = torch.tensor(df[label_col].values, dtype=torch.long)
        ds = HC3Dataset(texts, labels, tokenizer, max_length)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    train_loader = make_loader(train_df, shuffle=True)
    val_loader   = make_loader(val_df,   shuffle=False)
    test_loader  = make_loader(test_df,  shuffle=False)

    # ── Model ────────────────────────────────────────────────
    logger.info("Loading DistilBERT model …")
    model = DistilBertForSequenceClassification.from_pretrained(
        DISTILBERT_MODEL_NAME, num_labels=2
    )
    model = model.to(device)

    # ── Optimiser + scheduler ─────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=DISTILBERT_WEIGHT_DECAY,
    )
    total_steps   = len(train_loader) * epochs
    warmup_steps  = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # ── Training loop ─────────────────────────────────────────
    best_val_f1   = -1.0
    best_epoch    = -1
    history = []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_preds, val_probs, val_labels = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                labels_batch = batch["labels"]
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                logits = outputs.logits
                probs  = torch.softmax(logits, dim=-1)[:, 1]
                preds  = logits.argmax(dim=-1)
                val_preds.extend(preds.cpu().numpy())
                val_probs.extend(probs.cpu().numpy())
                val_labels.extend(labels_batch.numpy())

        val_preds  = np.array(val_preds)
        val_probs  = np.array(val_probs)
        val_labels = np.array(val_labels)
        val_m = compute_metrics(val_labels, val_preds, val_probs, f"BERT-val-e{epoch}")

        logger.info(
            "Epoch %d/%d | train_loss=%.4f | val_f1=%.4f | val_acc=%.4f",
            epoch, epochs, avg_train_loss, val_m["f1"], val_m["accuracy"],
        )
        history.append({"epoch": epoch, "train_loss": avg_train_loss, **val_m})

        # Early stopping: keep best
        if val_m["f1"] > best_val_f1:
            best_val_f1 = val_m["f1"]
            best_epoch  = epoch
            # Save best model
            ensure_dirs()
            model.save_pretrained(BERT_DIR / "best_model")
            tokenizer.save_pretrained(BERT_DIR / "best_model")

    train_time = round(time.time() - t0, 2)
    logger.info(
        "Training complete. Best val F1=%.4f at epoch %d. Time: %.1f s",
        best_val_f1, best_epoch, train_time,
    )

    # ── Load best model for test evaluation ──────────────────
    best_model = DistilBertForSequenceClassification.from_pretrained(
        BERT_DIR / "best_model"
    ).to(device)
    best_model.eval()

    test_preds, test_probs, test_labels = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            labels_batch = batch["labels"]
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = best_model(**batch)
            logits  = outputs.logits
            probs   = torch.softmax(logits, dim=-1)[:, 1]
            preds   = logits.argmax(dim=-1)
            test_preds.extend(preds.cpu().numpy())
            test_probs.extend(probs.cpu().numpy())
            test_labels.extend(labels_batch.numpy())

    y_test      = np.array(test_labels)
    y_test_pred = np.array(test_preds)
    y_test_prob = np.array(test_probs)

    test_metrics = compute_metrics(y_test, y_test_pred, y_test_prob, "DistilBERT")
    print_classification_report(y_test, y_test_pred, "DistilBERT (test)")

    # ── Plots ─────────────────────────────────────────────────
    plot_confusion_matrix(y_test, y_test_pred, "DistilBERT")
    plot_roc_curve(y_test, y_test_prob, "DistilBERT")
    plot_pr_curve(y_test, y_test_prob, "DistilBERT")

    # ── Save results ──────────────────────────────────────────
    save_metrics(test_metrics, "distilbert")
    save_predictions(test_df, y_test_pred, y_test_prob, "DistilBERT")

    params_out = {
        "model": "DistilBERT",
        "base_model": DISTILBERT_MODEL_NAME,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "batch_size": batch_size,
        "learning_rate": lr,
        "max_length": max_length,
        "best_val_f1": best_val_f1,
        "train_time_s": train_time,
    }
    with open(BERT_DIR / "best_params.json", "w") as fh:
        json.dump(params_out, fh, indent=2)

    pd.DataFrame(history).to_csv(BERT_DIR / "training_history.csv", index=False)

    return {
        "model_dir":    BERT_DIR / "best_model",
        "best_epoch":   best_epoch,
        "best_val_f1":  best_val_f1,
        "test_metrics": test_metrics,
        "history":      history,
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.data_loader import load_hc3, split_dataset
    from src.preprocessing import preprocess_dataframe

    logger.info("Loading and preprocessing HC3 …")
    df = load_hc3()
    df = preprocess_dataframe(df)
    train_df, val_df, test_df = split_dataset(df)

    results = train_distilbert(train_df, val_df, test_df)
    print(f"\n✅ DistilBERT training complete. Test F1: {results['test_metrics']['f1']}")
