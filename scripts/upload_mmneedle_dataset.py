#!/usr/bin/env python3
"""Upload the MMNeedle dataset assets to the Hugging Face Hub."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO_ID = "Wang-ML-Lab/MMNeedle"
DEFAULT_IMAGES = REPO_ROOT.parent / "mmneedle_gdrive" / "images_stitched.zip"
DEFAULT_METADATA = REPO_ROOT.parent / "mmneedle_gdrive" / "metadata_stitched.zip"
DEFAULT_CAPTIONS = REPO_ROOT.parent / "mmneedle_gdrive" / "file_to_caption.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Target dataset repo")
    parser.add_argument(
        "--images-zip",
        default=str(DEFAULT_IMAGES),
        help="Path to images_stitched.zip from Google Drive",
    )
    parser.add_argument(
        "--metadata-zip",
        default=str(DEFAULT_METADATA),
        help="Path to metadata_stitched.zip",
    )
    parser.add_argument(
        "--captions-json",
        default=str(DEFAULT_CAPTIONS),
        help="Path to file_to_caption.json",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face access token (defaults to HF_TOKEN env var)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the dataset repo as private (default: public)",
    )
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return path


def main() -> None:
    args = parse_args()
    if not args.hf_token:
        raise SystemExit("Set --hf-token or HF_TOKEN with a valid write token.")

    api = HfApi(token=args.hf_token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        exist_ok=True,
        private=args.private,
    )

    uploads = [
        (REPO_ROOT / "huggingface" / "mmneedle.py", "mmneedle.py"),
        (REPO_ROOT / "huggingface" / "README.md", "README.md"),
        (require_file(Path(args.images_zip)), "data/images_stitched.zip"),
        (require_file(Path(args.metadata_zip)), "data/metadata_stitched.zip"),
        (require_file(Path(args.captions_json)), "data/file_to_caption.json"),
    ]

    for local, remote in uploads:
        print(f"Uploading {local} -> {args.repo_id}:{remote}")
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=remote,
            repo_id=args.repo_id,
            repo_type="dataset",
        )

    print("Done.")


if __name__ == "__main__":
    main()
