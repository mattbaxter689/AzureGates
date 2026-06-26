import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import set_config
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging
import json
from pathlib import Path


set_config(transform_output="pandas")

logger = logging.getLogger(__name__)

TARGET_COL = "sleep_disorder_risk"


def infer_schema(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Auto-detect numeric vs categorical feature columns, excluding target"""
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    num_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df[feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()

    return num_cols, cat_cols


def load_drift_output(file_path: str) -> str:
    """
    Load drift output from Drift Detection step
    to determine model training needed
    """
    data_path = Path(file_path)
    drift_path = data_path / "drift.txt"

    if not drift_path.exists():
        raise FileNotFoundError(
            f"Drift output file not found in input path: {drift_path}"
        )
    logger.info("Drift output loaded successfully from mount")

    return drift_path.read_text().strip()


def load_final_run_output(file_path: str) -> str:
    """
    Load run id from final model run
    in model training step
    """
    data_path = Path(file_path)
    run_path = data_path / "run_id.txt"

    if not run_path.exists():
        raise FileNotFoundError(
            f"Run ID output file not found in input path: {run_path}"
        )
    logger.info("Final run id loaded successfully from mount")

    return run_path.read_text().strip()


def load_baseline_data(file_path: str) -> dict[str, float]:
    """
    Load baseline data uploaded from Data Versioning step
    as part of asset location
    """
    data_path = Path(file_path)
    baseline_path = data_path / "baseline.json"

    if not baseline_path.exists():
        raise FileNotFoundError(
            f"Baseline file not found in input path: {baseline_path}"
        )

    with open(baseline_path, "r") as f:
        baseline_stats = json.load(f)
    logger.info("Baseline stats successfully loaded from input mount.")

    return baseline_stats


def load_data(file_path: str) -> pd.DataFrame | tuple[pd.DataFrame, ...]:
    """
    Load data from an AzureML asset location.
    If the path is a folder, returns a tuple of DataFrames (one per CSV file).
    If the path is a file, returns a single DataFrame.
    """
    data_path = Path(file_path)

    if data_path.is_dir():
        order = ["train", "validation", "test"]
        files = {f.stem: f for f in data_path.glob("*.parquet")}
        return tuple(pd.read_parquet(files[name]) for name in order if name in files)

    return pd.read_csv(data_path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    1. Drop exact duplicate rows.
    2. Drop columns where >50% values are null.
    3. Impute remaining nulls (median for numeric, mode for categorical).
    4. Strip whitespace from string columns.
    """
    before = len(df)
    df = df.drop_duplicates().drop(columns=["person_id"])
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


def encode_features(
    df: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
    fit: bool = True,
    transformer: ColumnTransformer | None = None,
    label_encoder: LabelEncoder | None = None,
) -> tuple[pd.DataFrame, pd.Series, ColumnTransformer, LabelEncoder]:
    """
    Encode categorical columns with OneHotEncoder and scale numeric columns
    with MinMaxScaler.

    Returns the fitted scalers for transformation on validation and test data
    """
    data = df.drop(columns=[TARGET_COL])
    target = df[TARGET_COL]

    if fit:
        transformer = create_transform_pipeline(num_cols, cat_cols)
        label_encoder = LabelEncoder()
        data_tf = transformer.fit_transform(
            data,
        )
        target_tf = label_encoder.fit_transform(target)

    else:
        assert transformer is not None
        assert label_encoder is not None

        data_tf = transformer.transform(data)
        target_tf = label_encoder.transform(target)

    return data_tf, target_tf, transformer, label_encoder


def create_transform_pipeline(
    num_cols: list[str], cat_cols: list[str]
) -> ColumnTransformer:
    """
    Create the sklearn pipeline for transforming variables. Only for numerical and
    categorical data
    """

    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", MinMaxScaler())]
    )

    onehot_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("onehot", onehot_pipeline, cat_cols),
        ],
        remainder="drop",
    )
