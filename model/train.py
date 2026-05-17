"""
train.py — Training pipeline for the self-driving CNN.

Reads Udacity simulator driving_log.csv, splits into train/val,
runs augmented training with a Keras Sequence generator, saves model.h5.

Usage:
    python model/train.py
    python model/train.py --data data/driving_log.csv --epochs 20 --batch 32
"""

import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras

from model import build_model
from utils.data_generator import DataGenerator


# ── Config ────────────────────────────────────────────────────────────────────

DATA_CSV    = "data/driving_log.csv"
MODEL_OUT   = "model/model.h5"
EPOCHS      = 20
BATCH_SIZE  = 32
VAL_SPLIT   = 0.2
STEERING_CORRECTION = 0.25   # offset applied to left/right camera angles


# ── Load and expand driving log ───────────────────────────────────────────────

def load_samples(csv_path: str) -> list:
    """
    Parse driving_log.csv and expand each row into three samples
    (center, left, right camera) with appropriate steering correction.

    Returns list of (image_path, steering_angle) tuples.
    """
    df = pd.read_csv(csv_path, names=["center", "left", "right",
                                       "steering", "throttle", "brake", "speed"])
    samples = []
    for _, row in df.iterrows():
        angle = float(row["steering"])
        samples.append((row["center"].strip(), angle))
        samples.append((row["left"].strip(),   angle + STEERING_CORRECTION))
        samples.append((row["right"].strip(),  angle - STEERING_CORRECTION))

    print(f"Loaded {len(df)} rows → {len(samples)} samples (3× multi-camera expansion)")
    return samples


# ── Callbacks ─────────────────────────────────────────────────────────────────

def get_callbacks(model_path: str) -> list:
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        keras.callbacks.TensorBoard(log_dir="runs/", histogram_freq=0),
    ]


# ── Main training loop ────────────────────────────────────────────────────────

def train(args):
    print(f"\n{'='*55}")
    print("  Self-Driving Car — Behavioral Cloning Training")
    print(f"{'='*55}\n")

    # Load samples
    all_samples = load_samples(args.data)

    # Balance dataset: drastically reduce near-zero steering samples
    # (simulator produces many more straight-road frames than turns)
    straight = [s for s in all_samples if abs(s[1]) < 0.05]
    turning  = [s for s in all_samples if abs(s[1]) >= 0.05]
    # Keep 30% of straight samples to avoid straight-road bias
    straight_keep = straight[:int(len(straight) * 0.3)]
    balanced = straight_keep + turning
    np.random.shuffle(balanced)
    print(f"After balancing: {len(balanced)} samples "
          f"({len(straight_keep)} straight + {len(turning)} turning)")

    # Train/val split
    train_samples, val_samples = train_test_split(
        balanced, test_size=args.val_split, random_state=42)
    print(f"Train: {len(train_samples)} | Val: {len(val_samples)}\n")

    # Data generators
    train_gen = DataGenerator(train_samples, batch_size=args.batch, augment=True)
    val_gen   = DataGenerator(val_samples,   batch_size=args.batch, augment=False)

    # Build model
    model = build_model()
    model.summary()

    # Train
    print(f"\nTraining for up to {args.epochs} epochs (early stopping patience=5)...\n")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=get_callbacks(args.output),
        verbose=1,
    )

    # Summary
    best_val = min(history.history["val_loss"])
    print(f"\n✓ Training complete.")
    print(f"  Best val_loss : {best_val:.4f}")
    print(f"  Model saved   : {args.output}")
    print(f"  Run simulator autonomous mode with: python drive.py {args.output}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",      default=DATA_CSV,   help="Path to driving_log.csv")
    parser.add_argument("--output",    default=MODEL_OUT,  help="Where to save model.h5")
    parser.add_argument("--epochs",    type=int, default=EPOCHS)
    parser.add_argument("--batch",     type=int, default=BATCH_SIZE)
    parser.add_argument("--val-split", type=float, default=VAL_SPLIT)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    train(args)
