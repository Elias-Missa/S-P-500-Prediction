"""
MongoDB collection helpers for the S&P 500 prediction pipeline.

Provides idempotent upsert and read functions for each collection:
  - ohlcv:        raw daily prices and macro data
  - features:     engineered daily features
  - predictions:  model predictions per run
  - run_metadata: metadata per training run
"""

import datetime
import pandas as pd
import numpy as np
from pymongo import UpdateOne, ASCENDING, DESCENDING

from db import get_collection


# ---------------------------------------------------------------------------
# Index bootstrapping (called once on first write)
# ---------------------------------------------------------------------------

_indexes_ensured = set()


def _ensure_indexes(collection_name: str):
    """Create indexes if they don't already exist (idempotent)."""
    if collection_name in _indexes_ensured:
        return
    coll = get_collection(collection_name)

    if collection_name == "ohlcv":
        coll.create_index(
            [("symbol", ASCENDING), ("date", ASCENDING)],
            unique=True,
            name="symbol_date_unique",
        )
    elif collection_name == "features":
        coll.create_index(
            [("date", ASCENDING)],
            unique=True,
            name="date_unique",
        )
    elif collection_name == "predictions":
        coll.create_index(
            [("run_id", ASCENDING), ("date", ASCENDING)],
            unique=True,
            name="run_date_unique",
        )
        coll.create_index(
            [("run_id", ASCENDING)],
            name="run_id",
        )
    elif collection_name == "run_metadata":
        coll.create_index(
            [("run_id", ASCENDING)],
            unique=True,
            name="run_id_unique",
        )
        coll.create_index(
            [("created_at", DESCENDING)],
            name="created_at_desc",
        )

    _indexes_ensured.add(collection_name)


# ---------------------------------------------------------------------------
# Helpers: numpy / pandas types -> Python native (for BSON serialization)
# ---------------------------------------------------------------------------

def _sanitize_value(v):
    """Convert numpy/pandas scalars to Python-native types for MongoDB."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v) if np.isfinite(v) else None
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, pd.Timestamp):
        return v.to_pydatetime()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def _row_to_doc(date, row_series):
    """Convert a single pandas row (Series) into a MongoDB document."""
    doc = {"date": pd.Timestamp(date).to_pydatetime()}
    for col, val in row_series.items():
        doc[col] = _sanitize_value(val)
    return doc


# ---------------------------------------------------------------------------
# OHLCV collection
# ---------------------------------------------------------------------------

def upsert_ohlcv(df: pd.DataFrame):
    """
    Upsert raw OHLCV + macro data into the ``ohlcv`` collection.

    The DataFrame is expected to have a DatetimeIndex and columns like
    SPY, SPY_Volume, ^VIX, CL=F, etc.  Each column is stored as a
    separate *symbol* document keyed on (symbol, date).  This makes it
    easy to query "give me all SPY rows" or "all rows for 2024-01-15".

    Idempotent: re-running with the same data is safe (upsert).
    """
    _ensure_indexes("ohlcv")
    coll = get_collection("ohlcv")

    ops = []
    for date, row in df.iterrows():
        dt = pd.Timestamp(date).to_pydatetime()
        for col in df.columns:
            val = _sanitize_value(row[col])
            if val is None:
                continue
            ops.append(
                UpdateOne(
                    {"symbol": col, "date": dt},
                    {"$set": {"symbol": col, "date": dt, "value": val}},
                    upsert=True,
                )
            )
        if len(ops) >= 5000:
            coll.bulk_write(ops, ordered=False)
            ops = []

    if ops:
        coll.bulk_write(ops, ordered=False)

    print(f"[MongoDB] Upserted OHLCV data: {len(df)} dates, {len(df.columns)} symbols")


def load_ohlcv_df(symbols: list = None, since_date: str = None) -> pd.DataFrame:
    """
    Read OHLCV data from MongoDB back into a pivot DataFrame
    (DatetimeIndex, one column per symbol).
    """
    coll = get_collection("ohlcv")
    query = {}
    if symbols:
        query["symbol"] = {"$in": symbols}
    if since_date:
        query["date"] = {"$gte": pd.Timestamp(since_date).to_pydatetime()}

    cursor = coll.find(query, {"_id": 0})
    records = list(cursor)
    if not records:
        return pd.DataFrame()

    raw = pd.DataFrame(records)
    df = raw.pivot_table(index="date", columns="symbol", values="value")
    df.index = pd.DatetimeIndex(df.index)
    df.sort_index(inplace=True)
    return df


def get_last_ohlcv_date() -> datetime.date | None:
    """Return the most recent date stored in the ohlcv collection, or None."""
    coll = get_collection("ohlcv")
    doc = coll.find_one(sort=[("date", DESCENDING)], projection={"date": 1})
    if doc and "date" in doc:
        return pd.Timestamp(doc["date"]).date()
    return None


# ---------------------------------------------------------------------------
# Features collection
# ---------------------------------------------------------------------------

def upsert_features(df: pd.DataFrame):
    """
    Upsert engineered features into the ``features`` collection.

    One document per date.  All feature columns are stored as fields
    inside the document.

    Idempotent: re-running with the same data is safe (upsert).
    """
    _ensure_indexes("features")
    coll = get_collection("features")

    ops = []
    for date, row in df.iterrows():
        doc = _row_to_doc(date, row)
        ops.append(
            UpdateOne(
                {"date": doc["date"]},
                {"$set": doc},
                upsert=True,
            )
        )
        if len(ops) >= 2000:
            coll.bulk_write(ops, ordered=False)
            ops = []

    if ops:
        coll.bulk_write(ops, ordered=False)

    print(f"[MongoDB] Upserted features: {len(df)} dates, {len(df.columns)} columns")


def load_features_df(since_date: str = None) -> pd.DataFrame:
    """
    Read engineered features from MongoDB into a pandas DataFrame.

    Returns a DataFrame with DatetimeIndex — drop-in replacement for
    ``pd.read_csv(config.DATA_PATH, ...)``.
    """
    coll = get_collection("features")
    query = {}
    if since_date:
        query["date"] = {"$gte": pd.Timestamp(since_date).to_pydatetime()}

    cursor = coll.find(query, {"_id": 0}).sort("date", ASCENDING)
    records = list(cursor)
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df.set_index("date", inplace=True)
    df.index = pd.DatetimeIndex(df.index)
    df.sort_index(inplace=True)
    return df


def get_last_feature_date() -> datetime.date | None:
    """Return the most recent date in the features collection, or None."""
    coll = get_collection("features")
    doc = coll.find_one(sort=[("date", DESCENDING)], projection={"date": 1})
    if doc and "date" in doc:
        return pd.Timestamp(doc["date"]).date()
    return None


# ---------------------------------------------------------------------------
# Predictions collection
# ---------------------------------------------------------------------------

def save_predictions(run_id: str, df: pd.DataFrame, metadata: dict = None):
    """
    Save model predictions to the ``predictions`` collection and
    metadata to the ``run_metadata`` collection.

    Parameters
    ----------
    run_id : str
        Unique identifier for this training run (e.g. folder name).
    df : pd.DataFrame
        Predictions DataFrame.  Must have a DatetimeIndex and at least
        a ``y_pred`` column (additional columns like ``y_true`` are fine).
    metadata : dict, optional
        Run-level metadata (model type, config snapshot, aggregate metrics).
    """
    # --- predictions ---
    _ensure_indexes("predictions")
    coll = get_collection("predictions")

    ops = []
    for date, row in df.iterrows():
        doc = _row_to_doc(date, row)
        doc["run_id"] = run_id
        ops.append(
            UpdateOne(
                {"run_id": run_id, "date": doc["date"]},
                {"$set": doc},
                upsert=True,
            )
        )
        if len(ops) >= 2000:
            coll.bulk_write(ops, ordered=False)
            ops = []
    if ops:
        coll.bulk_write(ops, ordered=False)

    print(f"[MongoDB] Saved {len(df)} prediction rows for run '{run_id}'")

    # --- run_metadata ---
    if metadata is not None:
        _ensure_indexes("run_metadata")
        meta_coll = get_collection("run_metadata")
        meta_doc = {k: _sanitize_value(v) for k, v in metadata.items()}
        meta_doc["run_id"] = run_id
        meta_doc.setdefault("created_at", datetime.datetime.utcnow())
        meta_coll.update_one(
            {"run_id": run_id},
            {"$set": meta_doc},
            upsert=True,
        )
        print(f"[MongoDB] Saved run metadata for '{run_id}'")


def load_predictions(run_id: str) -> pd.DataFrame:
    """Load predictions for a specific run into a DataFrame."""
    coll = get_collection("predictions")
    cursor = coll.find({"run_id": run_id}, {"_id": 0, "run_id": 0}).sort("date", ASCENDING)
    records = list(cursor)
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df.set_index("date", inplace=True)
    df.index = pd.DatetimeIndex(df.index)
    return df


def load_latest_predictions() -> tuple[pd.DataFrame, dict | None]:
    """
    Load predictions and metadata for the most recent training run.

    Returns (predictions_df, metadata_dict) or (empty_df, None).
    """
    meta_coll = get_collection("run_metadata")
    latest = meta_coll.find_one(sort=[("created_at", DESCENDING)])
    if not latest:
        return pd.DataFrame(), None

    run_id = latest["run_id"]
    df = load_predictions(run_id)
    latest.pop("_id", None)
    return df, latest


def load_run_metadata(run_id: str) -> dict | None:
    """Load metadata for a specific run."""
    coll = get_collection("run_metadata")
    doc = coll.find_one({"run_id": run_id}, {"_id": 0})
    return doc
