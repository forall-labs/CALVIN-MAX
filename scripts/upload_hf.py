"""Upload CALVIN-MAX data to a private Hugging Face dataset repo (D-036). Run on the Mac after `hf auth login`.

    python3.10 scripts/upload_hf.py --repo calvin-max            # -> <your-username>/calvin-max (private)

Uploads:
  benchmark/  frozen chains, manifests (versioned)
  results/    calibration, base/<policy>/{summary.json,traces/*.pkl}, cases/<run>/{results.jsonl,summary.json,
              behaviour.jsonl,traces/*.pkl}, merged summaries
  README.md   dataset card (scripts/hf_dataset_card.md)
upload_large_folder is resumable: rerun to continue after an interruption, or to add new runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="calvin-max")
    ap.add_argument("--public", action="store_true", help="default is private")
    args = ap.parse_args()
    api = HfApi()
    user = api.whoami()["name"]
    repo_id = f"{user}/{args.repo}"
    api.create_repo(repo_id, repo_type="dataset", private=not args.public, exist_ok=True)

    api.upload_file(path_or_fileobj=str(ROOT / "scripts" / "hf_dataset_card.md"), path_in_repo="README.md",
                    repo_id=repo_id, repo_type="dataset", commit_message="Dataset card")
    print(f"uploading benchmark/ and results/ -> https://huggingface.co/datasets/{repo_id} (private={not args.public})")
    api.upload_large_folder(
        repo_id=repo_id, repo_type="dataset", folder_path=str(ROOT),
        allow_patterns=["benchmark/**", "results/**"],
        ignore_patterns=["results/.mirror/**", "results/.backup/**", "**/.DS_Store", "**/*.tgz"],
    )
    print("done:", f"https://huggingface.co/datasets/{repo_id}")

if __name__ == "__main__":
    main()
