"""Read/write utilities for Parquet and CSV files."""

from pathlib import Path
from typing import Optional

import pandas as pd


def save_parquet(df: pd.DataFrame, path: Path, **kwargs) -> Path:
    """Save a DataFrame to Parquet format, creating parent directories if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow", **kwargs)
    return path


def load_parquet(path: Path) -> pd.DataFrame:
    """Load a Parquet file into a DataFrame."""
    return pd.read_parquet(path, engine="pyarrow")


def save_csv(df: pd.DataFrame, path: Path, **kwargs) -> Path:
    """Save a DataFrame to CSV, creating parent directories if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, **kwargs)
    return path


def load_csv(path: Path, **kwargs) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    return pd.read_csv(path, **kwargs)


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist. Returns the path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_parquet_files(directory: Path) -> list[Path]:
    """List all .parquet files in a directory."""
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted(directory.glob("*.parquet"))
