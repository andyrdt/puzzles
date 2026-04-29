"""
Push a local checkpoint directory to HuggingFace Hub.

Uploads model.pt, config.json, and model.py (so notebooks can import directly).

Usage:
    python 05_2026/push_to_hf.py --local_dir 05_2026/puzzle1/checkpoints --repo_id andyrdt/05_2026_puzzle_1
"""

import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, upload_folder


def main():
    p = argparse.ArgumentParser(description="Push checkpoint to HuggingFace Hub")
    p.add_argument("--local_dir", type=str, required=True, help="Path to local checkpoint dir (must contain model.pt and config.json)")
    p.add_argument("--repo_id", type=str, required=True, help="HuggingFace repo id, e.g. andyrdt/05_2026_puzzle_1")
    p.add_argument("--private", action="store_true", help="Create as private repo")
    args = p.parse_args()

    local_dir = Path(args.local_dir)
    assert (local_dir / "model.pt").exists(), f"model.pt not found in {local_dir}"
    assert (local_dir / "config.json").exists(), f"config.json not found in {local_dir}"

    # model.py lives next to this script
    model_py = Path(__file__).resolve().parent / "model.py"
    assert model_py.exists(), f"model.py not found at {model_py}"

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shutil.copy(local_dir / "model.pt", tmp / "model.pt")
        shutil.copy(local_dir / "config.json", tmp / "config.json")
        shutil.copy(model_py, tmp / "model.py")

        api = HfApi()
        api.create_repo(args.repo_id, exist_ok=True, private=args.private)
        upload_folder(repo_id=args.repo_id, folder_path=str(tmp))
        print(f"Pushed to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
