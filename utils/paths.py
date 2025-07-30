# a utility file to handle file paths for experiment results, per huggingface base model ID

import os
from huggingface_hub import HfFileSystem

top_level_dir = "results"
subdirs: list[str] = ["activations", "attack_results", "cautious_dir", "dataset"]

fs: HfFileSystem = HfFileSystem()

def validate_hf_id(hf_id: str):
    # first validate two parts separeted by /
    assert(len(hf_id.split("/")) == 2), "Hugging Face ID should be in the format 'org_name/model_name'."
    assert fs.exists(hf_id), f"Hugging Face ID '{hf_id}' does not exist on the hub."

def make_dirs(hf_id: str) -> bool:
    validate_hf_id(hf_id)
    hf_path: str = os.path.join(hf_id.split("/")[0], hf_id.split("/")[1])

    for subdir in subdirs:
        dir_path: str = os.path.join(top_level_dir, hf_path, subdir)
        # create directory if it does not exist
        if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path)
            except OSError as e:
                print(f"Error creating directory {dir_path}: {e}")
                return False
    return True


def get_path(hf_id: str, subdir: str) -> str:
    hf_path = os.path.join(hf_id.split("/")[0], hf_id.split("/")[1])
    return os.path.join(top_level_dir, hf_path, subdir)