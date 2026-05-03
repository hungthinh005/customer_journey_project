"""
Download the Online Retail II dataset from Kaggle using kagglehub.
Copies the raw file into data/raw/ for reproducibility.
"""

import shutil
from pathlib import Path

import kagglehub

from config import DATASET_NAME, DATA_RAW_DIR


def download_dataset():
    """Download and copy the Online Retail II dataset."""
    print(f"Downloading dataset: {DATASET_NAME}")
    path = kagglehub.dataset_download(DATASET_NAME)
    print(f"Downloaded to: {path}")

    # Copy all files to data/raw/
    src_path = Path(path)
    for file in src_path.rglob("*"):
        if file.is_file():
            dest = DATA_RAW_DIR / file.name
            shutil.copy2(file, dest)
            print(f"  Copied: {file.name} -> {dest}")

    print(f"\nDataset ready in: {DATA_RAW_DIR}")
    return DATA_RAW_DIR


if __name__ == "__main__":
    download_dataset()
