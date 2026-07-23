from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tabular_synth.data import (  # noqa: E402
    ADULT_COLUMNS,
    HELOC_COLUMNS,
    load_adult,
    load_heloc,
)


ADULT_URLS = {
    "adult.data": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    "adult.test": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
    "adult.names": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names",
}
HELOC_URL = "https://huggingface.co/datasets/mstz/heloc/resolve/main/risk/train.csv"


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.dataset in {"adult", "all"}:
        adult_summary = download_adult(outdir, force=args.force)
        print(json.dumps(adult_summary, indent=2, sort_keys=True))

    if args.dataset in {"heloc", "all"}:
        heloc_summary = download_heloc(outdir, force=args.force)
        print(json.dumps(heloc_summary, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Adult and HELOC datasets and normalize them for experiments."
    )
    parser.add_argument("--dataset", choices=["adult", "heloc", "all"], required=True)
    parser.add_argument("--outdir", default="data")
    parser.add_argument("--force", action="store_true", help="Re-download and overwrite files.")
    return parser.parse_args()


def download_adult(outdir: Path, force: bool = False) -> dict:
    raw_dir = outdir / "raw" / "adult"
    raw_dir.mkdir(parents=True, exist_ok=True)
    downloaded = {}
    for filename, url in ADULT_URLS.items():
        destination = raw_dir / filename
        download_url(url, destination, force=force)
        downloaded[filename] = str(destination)

    train = load_adult(raw_dir / "adult.data")
    test = load_adult(raw_dir / "adult.test")
    combined = pd.concat([train, test], ignore_index=True)

    train_path = outdir / "adult_train.csv"
    test_path = outdir / "adult_test.csv"
    combined_path = outdir / "adult.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    combined.to_csv(combined_path, index=False)

    validate_adult(combined_path)
    summary = {
        "dataset": "adult",
        "source": ADULT_URLS,
        "raw_files": downloaded,
        "train_path": str(train_path),
        "test_path": str(test_path),
        "combined_path": str(combined_path),
        "n_train": len(train),
        "n_test": len(test),
        "n_combined": len(combined),
        "columns": ADULT_COLUMNS,
    }
    write_manifest(outdir / "adult_manifest.json", summary)
    return summary


def download_heloc(outdir: Path, force: bool = False) -> dict:
    raw_dir = outdir / "raw" / "heloc"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / "mstz_heloc_risk_train.csv"
    download_url(HELOC_URL, raw_path, force=force)

    heloc = load_heloc(raw_path)
    output_path = outdir / "heloc_dataset_v1.csv"
    heloc.to_csv(output_path, index=False)

    validate_heloc(output_path)
    summary = {
        "dataset": "heloc",
        "source": HELOC_URL,
        "raw_path": str(raw_path),
        "path": str(output_path),
        "n_rows": len(heloc),
        "columns": HELOC_COLUMNS,
        "note": "HuggingFace mstz/heloc columns are normalized to canonical FICO HELOC names.",
    }
    write_manifest(outdir / "heloc_manifest.json", summary)
    return summary


def download_url(url: str, destination: Path, force: bool = False) -> None:
    if destination.exists() and not force:
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".tmp")
    request = urllib.request.Request(url, headers={"User-Agent": "tabular-llm-synth/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        tmp_path.write_bytes(response.read())
    tmp_path.replace(destination)


def validate_adult(path: Path) -> None:
    df = load_adult(path)
    if df.empty:
        raise ValueError(f"Downloaded Adult dataset is empty: {path}")
    if list(df.columns) != ADULT_COLUMNS:
        raise ValueError(f"Unexpected Adult columns in {path}: {list(df.columns)}")


def validate_heloc(path: Path) -> None:
    df = load_heloc(path)
    if df.empty:
        raise ValueError(f"Downloaded HELOC dataset is empty: {path}")
    if list(df.columns) != HELOC_COLUMNS:
        raise ValueError(f"Unexpected HELOC columns in {path}: {list(df.columns)}")


def write_manifest(path: Path, summary: dict) -> None:
    path.write_text(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
