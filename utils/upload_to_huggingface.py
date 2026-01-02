"""
Script to upload orthogonalized models to Hugging Face Hub.

Usage:
    uv run -m utils.upload_to_huggingface \
        --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
        --baseline_layer 11 \
        --cot_layer 17 \
        --hf_username your_username

Note: Requires HF_TOKEN in .env file or as environment variable.
"""

import argparse
import os

from dotenv import load_dotenv
from huggingface_hub import HfApi, login

# Load environment variables from .env file
load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Upload orthogonalized models to Hugging Face Hub"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Original model name (e.g., 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B')",
    )
    parser.add_argument(
        "--baseline_layer",
        type=int,
        required=True,
        help="Baseline layer number (e.g., 11)",
    )
    parser.add_argument(
        "--cot_layer",
        type=int,
        required=True,
        help="CoT layer number (e.g., 17)",
    )
    parser.add_argument(
        "--hf_username",
        type=str,
        required=True,
        help="Your Hugging Face username or organization name",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Base results directory (default: 'results')",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the uploaded repositories private",
    )
    parser.add_argument(
        "--skip_baseline",
        action="store_true",
        help="Skip uploading the baseline model",
    )
    parser.add_argument(
        "--skip_cot",
        action="store_true",
        help="Skip uploading the CoT model",
    )
    return parser.parse_args()


def get_repo_name_from_model(model_name: str) -> str:
    """Extract a clean repo name from the model name."""
    # Replace '/' with '-' and clean up
    return model_name.replace("/", "-")


def upload_model(
    api: HfApi,
    local_dir: str,
    repo_id: str,
    model_type: str,
    layer: int,
    private: bool = False,
):
    """Upload a model directory to Hugging Face Hub."""
    if not os.path.exists(local_dir):
        print(f"Error: Directory not found: {local_dir}")
        return False

    print(f"\nUploading {model_type} model (layer {layer})...")
    print(f"  Local directory: {local_dir}")
    print(f"  Repository: {repo_id}")

    # Create the repository if it doesn't exist
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=private,
            exist_ok=True,
        )
        print(f"  Repository created/verified: {repo_id}")
    except Exception as e:
        print(f"  Error creating repository: {e}")
        return False

    # Upload all files in the directory
    try:
        api.upload_folder(
            folder_path=local_dir,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload orthogonalized {model_type} model (layer {layer})",
        )
        print(f"  Successfully uploaded to: https://huggingface.co/{repo_id}")
        return True
    except Exception as e:
        print(f"  Error uploading: {e}")
        return False


def create_model_card(
    api: HfApi,
    repo_id: str,
    original_model: str,
    model_type: str,
    layer: int,
):
    """Create a model card README for the uploaded model."""
    model_card = f"""---
license: apache-2.0
base_model: {original_model}
tags:
  - orthogonalized
  - {model_type}
  - layer-{layer}
---

# Orthogonalized {model_type.title()} Model (Layer {layer})

This model is an orthogonalized version of [{original_model}](https://huggingface.co/{original_model}).

## Model Details

- **Base Model:** {original_model}
- **Model Type:** {model_type.title()}
- **Orthogonalization Layer:** {layer}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
```

## Citation

If you use this model, please cite the original model and the orthogonalization method used.
"""

    try:
        api.upload_file(
            path_or_fileobj=model_card.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add model card",
        )
        print(f"  Model card created for {repo_id}")
    except Exception as e:
        print(f"  Warning: Could not create model card: {e}")


def main():
    args = parse_args()

    # Login to Hugging Face using HF_TOKEN environment variable
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Error: HF_TOKEN not found. Set it in .env file or as environment variable.")
        return 1
    
    login(token=hf_token)

    api = HfApi()

    # Construct paths and repo names
    clean_model_name = get_repo_name_from_model(args.model_name)

    baseline_dir = os.path.join(
        args.results_dir,
        args.model_name,
        f"ortho_model_baseline_layer_{args.baseline_layer}",
    )
    cot_dir = os.path.join(
        args.results_dir,
        args.model_name,
        f"ortho_model_cot_layer_{args.cot_layer}",
    )

    baseline_repo_id = f"{args.hf_username}/{clean_model_name}-ortho-baseline-layer-{args.baseline_layer}"
    cot_repo_id = f"{args.hf_username}/{clean_model_name}-ortho-cot-layer-{args.cot_layer}"

    print("=" * 60)
    print("Hugging Face Model Upload Script")
    print("=" * 60)
    print(f"Original model: {args.model_name}")
    print(f"Results directory: {args.results_dir}")
    print(f"Private repos: {args.private}")

    success_count = 0
    total_count = 0

    # Upload baseline model
    if not args.skip_baseline:
        total_count += 1
        if upload_model(
            api,
            baseline_dir,
            baseline_repo_id,
            "baseline",
            args.baseline_layer,
            args.private,
        ):
            create_model_card(
                api, baseline_repo_id, args.model_name, "baseline", args.baseline_layer
            )
            success_count += 1

    # Upload CoT model
    if not args.skip_cot:
        total_count += 1
        if upload_model(
            api,
            cot_dir,
            cot_repo_id,
            "cot",
            args.cot_layer,
            args.private,
        ):
            create_model_card(api, cot_repo_id, args.model_name, "cot", args.cot_layer)
            success_count += 1

    print("\n" + "=" * 60)
    print(f"Upload complete: {success_count}/{total_count} models uploaded successfully")
    print("=" * 60)

    if success_count == total_count:
        print("\nAll models uploaded successfully!")
        if not args.skip_baseline:
            print(f"  Baseline: https://huggingface.co/{baseline_repo_id}")
        if not args.skip_cot:
            print(f"  CoT: https://huggingface.co/{cot_repo_id}")
    else:
        print("\nSome uploads failed. Please check the error messages above.")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())