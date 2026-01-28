import argparse
from huggingface_hub import HfApi


api = HfApi()

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Upload a large folder to HuggingFace Hub")
    parser.add_argument("--repo_id", type=str, help="Repository ID (e.g., 'username/repo-name')")
    parser.add_argument("--repo_type", type=str, help="Repository type (e.g., 'dataset', 'model', 'space')")
    parser.add_argument("--folder_path", type=str, help="Local path to the folder to upload")

    args = parser.parse_args()

    api.upload_large_folder(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        folder_path=args.folder_path,
    )