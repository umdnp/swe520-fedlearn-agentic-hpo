"""
Compute model metadata and save the fitted preprocessor.

This script:
  - Reconstructs the federated client partitions
  - Applies the same 60/20/20 train/val/test per client
  - Unions all client-local training splits
  - Fits the shared preprocessor ONLY on that union
  - Computes:
      - n_features (after preprocessing)
      - classes (unique values of prolonged_stay from training data)
      - intercept (zero vector of length n_classes)
  - Saves:
      - configs/model_meta.json
      - configs/preprocessor.pkl

Run:
    python compute_model_metadata.py
"""

import json
from pathlib import Path

import joblib
import numpy as np

from fedlearn.common.data_split import get_client_train_union
from fedlearn.common.preprocessing import build_preprocessor

# Constants

PROJECT_ROOT = Path(__file__).resolve().parents[3]

CONFIG_DIR = PROJECT_ROOT / "configs"
META_PATH = CONFIG_DIR / "model_meta.json"
PREPROC_PATH = CONFIG_DIR / "preprocessor.pkl"


def main():
    # Ensure config directory exists
    if not CONFIG_DIR.exists():
        print(f"Config directory '{CONFIG_DIR}' does not exist, creating it ...")
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading union of federated client training data ...")
    X_train, y_train = get_client_train_union()

    print(f"count = {len(X_train):,}")

    print("Fitting preprocessing pipeline ...")
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)

    print("Transforming training data to compute feature dimension ...")
    X_proc = preprocessor.transform(X_train)
    n_features = X_proc.shape[1]

    # compute classes from y
    classes = np.unique(y_train)
    classes_list = [int(c) for c in classes]

    # create initial intercept as zeros, one per class
    intercept = [0.0] * len(classes_list)

    print("\n--------------------------------------------")
    print(f"Computed feature dimension: {n_features}")
    print(f"Classes: {classes_list}")
    print(f"Initial intercept (zeros): {intercept}")
    print("--------------------------------------------\n")

    meta = {
        "n_features": int(n_features),
        "classes": classes_list,
        "intercept": intercept,
    }

    print(f"Saving model metadata to {META_PATH}")
    with META_PATH.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saving fitted preprocessor to {PREPROC_PATH}")
    joblib.dump(preprocessor, PREPROC_PATH)

    print("Done!")


if __name__ == "__main__":
    main()
