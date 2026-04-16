"""Data ingestion: download, validate, and persist raw datasets."""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)


class DataIngestionError(Exception):
    """Raised when data ingestion fails validation or download."""


def download_file(
    url: str,
    destination: Path,
    expected_sha256: Optional[str] = None,
    chunk_size: int = 8192,
    timeout: int = 60,
) -> Path:
    """Download a file from *url* to *destination*, optionally verifying its checksum.

    Args:
        url: Remote URL to download from.
        destination: Local path to write the file to.
        expected_sha256: If provided, the downloaded file's SHA-256 must match.
        chunk_size: Number of bytes per streaming chunk.
        timeout: HTTP request timeout in seconds.

    Returns:
        The resolved destination path.

    Raises:
        DataIngestionError: If the download fails or checksum mismatches.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading %s → %s", url, destination)
    try:
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise DataIngestionError(f"Failed to download {url}: {exc}") from exc

    sha256 = hashlib.sha256()
    with destination.open("wb") as fh:
        for chunk in response.iter_content(chunk_size=chunk_size):
            fh.write(chunk)
            sha256.update(chunk)

    if expected_sha256 is not None:
        actual = sha256.hexdigest()
        if actual != expected_sha256.lower():
            destination.unlink(missing_ok=True)
            raise DataIngestionError(
                f"SHA-256 mismatch for {destination.name}: "
                f"expected={expected_sha256}, got={actual}"
            )
        logger.info("Checksum verified for %s", destination.name)

    logger.info("Downloaded %s (%.2f MB)", destination.name, destination.stat().st_size / 1e6)
    return destination


def load_csv(
    path: Path,
    expected_columns: Optional[list[str]] = None,
    expected_min_rows: int = 1,
) -> pd.DataFrame:
    """Load a CSV file and run basic structural validation.

    Args:
        path: Path to the CSV file.
        expected_columns: If provided, raise if any column is missing.
        expected_min_rows: Minimum acceptable number of rows after loading.

    Returns:
        Validated DataFrame.

    Raises:
        DataIngestionError: If the file is missing required columns or is too small.
    """
    path = Path(path)
    if not path.exists():
        raise DataIngestionError(f"CSV file not found: {path}")

    df = pd.read_csv(path)
    logger.info("Loaded %s — shape=%s", path.name, df.shape)

    if expected_columns:
        missing = set(expected_columns) - set(df.columns)
        if missing:
            raise DataIngestionError(
                f"Missing required columns in {path.name}: {sorted(missing)}"
            )

    if len(df) < expected_min_rows:
        raise DataIngestionError(
            f"{path.name} has {len(df)} rows; expected at least {expected_min_rows}"
        )

    return df


def load_parquet(
    path: Path,
    expected_columns: Optional[list[str]] = None,
    expected_min_rows: int = 1,
) -> pd.DataFrame:
    """Load a Parquet file and run basic structural validation.

    Args:
        path: Path to the Parquet file.
        expected_columns: If provided, raise if any column is missing.
        expected_min_rows: Minimum acceptable number of rows after loading.

    Returns:
        Validated DataFrame.

    Raises:
        DataIngestionError: If the file is missing required columns or is too small.
    """
    path = Path(path)
    if not path.exists():
        raise DataIngestionError(f"Parquet file not found: {path}")

    df = pd.read_parquet(path)
    logger.info("Loaded %s — shape=%s", path.name, df.shape)

    if expected_columns:
        missing = set(expected_columns) - set(df.columns)
        if missing:
            raise DataIngestionError(
                f"Missing required columns in {path.name}: {sorted(missing)}"
            )

    if len(df) < expected_min_rows:
        raise DataIngestionError(
            f"{path.name} has {len(df)} rows; expected at least {expected_min_rows}"
        )

    return df


def get_raw_data_dir() -> Path:
    """Return the raw data directory, resolving from the env var RAW_DATA_DIR or default."""
    raw_dir = os.environ.get("RAW_DATA_DIR", "data/raw")
    path = Path(raw_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_processed_data_dir() -> Path:
    """Return the processed data directory, resolving from env var or default."""
    proc_dir = os.environ.get("PROCESSED_DATA_DIR", "data/processed")
    path = Path(proc_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path
