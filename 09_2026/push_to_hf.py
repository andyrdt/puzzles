"""Upload the September model artifacts to Hugging Face Hub.

This script performs no action unless it is invoked explicitly.
"""

import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, upload_folder
from huggingface_hub.utils import HfHubHTTPError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local_dir",
        type=Path,
        required=True,
        help="checkpoint directory containing model.pt and config.json",
    )
    parser.add_argument(
        "--repo_id",
        required=True,
        help="destination such as andyrdt/09_2026_puzzle_1",
    )
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    checkpoint = args.local_dir.resolve()
    model_path = checkpoint / "model.pt"
    config_path = checkpoint / "config.json"
    for path in (model_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    model_source = Path(__file__).resolve().parent / "model.py"
    with tempfile.TemporaryDirectory() as temporary_directory:
        upload_directory = Path(temporary_directory)
        shutil.copy2(model_path, upload_directory / "model.pt")
        shutil.copy2(config_path, upload_directory / "config.json")
        shutil.copy2(model_source, upload_directory / "model.py")

        api = HfApi()
        try:
            api.repo_info(args.repo_id, repo_type="model")
        except HfHubHTTPError as error:
            if error.response.status_code != 404:
                raise
            api.create_repo(args.repo_id, private=args.private)
        upload_folder(repo_id=args.repo_id, folder_path=upload_directory)

    print(f"Uploaded to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
