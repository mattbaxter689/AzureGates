import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import set_config
import logging

set_config(transform_output="pandas")

logger = logging.getLogger(__name__)

TARGET_COL = "sleep_disorder_risk"


def infer_schema(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Auto-detect numeric vs categorical feature columns, excluding target"""
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    num_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df[feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()

    return num_cols, cat_cols


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    1. Drop exact duplicate rows.
    2. Drop columns where >50% values are null.
    3. Impute remaining nulls (median for numeric, mode for categorical).
    4. Strip whitespace from string columns.
    """
    before = len(df)
    df = df.drop_duplicates()
    logger.info(f"Dropped {before - len(df)} duplicate rows")

    null_frac = df.isnull().mean()
    high_null_cols = null_frac[null_frac > 0.5].index.tolist()
    if high_null_cols:
        logger.warning(f"Dropping high null columns: {high_null_cols}")
        df = df.drop(columns=high_null_cols)

    num_cols, cat_cols = infer_schema(df)

    for col in num_cols:
        median = df[col].median()
        n_filled = df[col].isnull().sum()
        if n_filled:
            df[col] = df[col].fillna(median)
            logger.debug(f"Imputed {n_filled} nulls in {col} with median={median}")

    for col in cat_cols:
        if col == TARGET_COL:
            continue
        mode_val = df[col].mode(dropna=True)
        if len(mode_val):
            fill_value = mode_val.iloc[0]
            df[col] = df[col].fillna(fill_value)

        df[col] = df[col].astype(str).str.strip()

    return df.reset_index(drop=True)


def split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified train / val / test split.
    test_frac = 1 - train_frac - val_frac.
    """
    assert 0 < train_frac < 1
    assert 0 < val_frac < 1
    assert train_frac + val_frac < 1

    test_frac = 1.0 - train_frac - val_frac

    stratify = df[TARGET_COL] if TARGET_COL in df.columns else None

    train_df, temp_df = train_test_split(
        df,
        test_size=(val_frac + test_frac),
        random_state=seed,
        stratify=stratify,
    )
    relative_test = test_frac / (val_frac + test_frac)
    stratify_temp = temp_df[TARGET_COL] if stratify is not None else None
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test,
        random_state=seed,
        stratify=stratify_temp,
    )

    logger.info(f"Split -> train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(val_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)
    return train_df, val_df, test_df


def compute_baseline_stats(df: pd.DataFrame, num_cols: list[str]) -> dict:
    """
    Compute baseline distribution statistics used by the drift detection gate.
    Returns a dict of {col: {mean, std, min, max, p25, p50, p75}}.
    """
    stats: dict = {}
    for col in num_cols:
        desc = df[col].describe()
        stats[col] = {
            "mean": float(desc["mean"]),
            "std": float(desc["std"]),
            "min": float(desc["min"]),
            "p25": float(desc["25%"]),
            "p50": float(desc["50%"]),
            "p75": float(desc["75%"]),
            "max": float(desc["max"]),
            "n": int(desc["count"]),
        }
    return stats
