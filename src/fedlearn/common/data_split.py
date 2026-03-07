from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from fedlearn.common.annotation import annotate_categorical_columns
from fedlearn.common.preprocessing import ALL_FEATURES

# Constants

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DUCKDB_PATH = PROJECT_ROOT / "data" / "duckdb" / "fedlearn.duckdb"
VIEW_NAME = "v_features_icu_stay_clean"

TARGET_COL = "prolonged_stay"
REGION_COL = "hospital_region"

CLIENT_REGION_MAP: dict[str, list[str | None]] = {
    "client_midwest": ["Midwest"],
    "client_south": ["South"],
    "client_other": ["West", "Northeast", None],
}

# Partitioning scheme
# partition-id = 0 -> client_midwest
# partition-id = 1 -> client_south
# partition-id = 2 -> client_other
CLIENT_KEYS = ("client_midwest", "client_south", "client_other")

SPLIT_RANDOM_STATE = 42
EVAL_SIZE = 0.2


def load_client_partition(client_key: str) -> pd.DataFrame:
    """
    Load only this client's partition from DuckDB.

    The mapping is:
      partition-id -> client bucket -> list of raw regions.

    Example:
      partition-id=0 -> "client_midwest" -> ["Midwest"]
      partition-id=1 -> "client_south"   -> ["South"]
      partition-id=2 -> "client_other"   -> ["West", "Northeast", NULL]
    """
    if client_key not in CLIENT_REGION_MAP:
        raise KeyError(f"Unknown client key: {client_key!r}")

    regions = CLIENT_REGION_MAP[client_key]

    conn = duckdb.connect(DUCKDB_PATH, read_only=True)
    try:
        include_null = None in regions
        real_regions = [r for r in regions if r is not None]

        where_clauses = []
        params: list[str] = []

        if real_regions:
            placeholders = ", ".join(["?"] * len(real_regions))
            where_clauses.append(f"{REGION_COL} IN ({placeholders})")
            params.extend(real_regions)

        if include_null:
            where_clauses.append(f"{REGION_COL} IS NULL")

        where_sql = " OR ".join(where_clauses) if where_clauses else "TRUE"

        # noinspection SqlNoDataSourceInspection
        query = f"SELECT * FROM {VIEW_NAME} WHERE {where_sql} ORDER BY patientunitstayid"

        df = conn.execute(query, params).df()
    finally:
        conn.close()

    # normalize pandas.NA -> np.nan so sklearn imputers are happy
    df = df.where(df.notna(), np.nan)

    # ensure categorical columns have right categories
    df = annotate_categorical_columns(df)

    return df


def _split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Apply the exact same feature selection and 80/20 split used by clients.
    """
    if df.empty:
        raise RuntimeError("No rows found for partition")

    y = df[TARGET_COL]

    feat_cols = list(ALL_FEATURES)
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Partition is missing expected feature columns: {missing}")

    X = df[feat_cols]

    X_train, X_eval, y_train, y_eval = train_test_split(
        X,
        y,
        test_size=EVAL_SIZE,
        random_state=SPLIT_RANDOM_STATE,
        stratify=y if y.nunique() > 1 else None,
    )

    return X_train, y_train, X_eval, y_eval


def get_client_train_eval_by_key(client_key: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Return local train/eval split for one logical client.
    """
    return _split_xy(load_client_partition(client_key))


def get_client_train_union() -> tuple[pd.DataFrame, pd.Series]:
    """
    Union all client-local training splits into one shared training set.
    """
    X_parts: list[pd.DataFrame] = []
    y_parts: list[pd.Series] = []

    for client_key in CLIENT_KEYS:
        X_train, y_train, _, _ = get_client_train_eval_by_key(client_key)
        X_parts.append(X_train)
        y_parts.append(y_train)

    X_train_all = pd.concat(X_parts, axis=0, ignore_index=True)
    y_train_all = pd.concat(y_parts, axis=0, ignore_index=True)

    return X_train_all, y_train_all
