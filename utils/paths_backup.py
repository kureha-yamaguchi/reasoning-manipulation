# a utility file to handle file paths for experiment results, per huggingface base model ID

import os
import json
from datetime import datetime
from huggingface_hub import HfFileSystem, model_info
from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError

top_level_dir = "results"
subdirs: list[str] = ["activations", "attack_results", "cautious_dir", "dataset"]

fs: HfFileSystem = HfFileSystem()

def validate_hf_id(hf_id: str):
    # first validate two parts separeted by /
    assert(len(hf_id.split("/")) == 2), "Hugging Face ID should be in the format 'org_name/model_name'."
    assert fs.exists(hf_id), f"Hugging Face ID '{hf_id}' does not exist on the hub."

def get_model_info(hf_id: str) -> dict:
    """Retrieve model information from Hugging Face Hub."""
    try:
        info = model_info(hf_id)
        
        # Get basic info from direct attributes
        model_data = {
            "model_id": info.id,
            "model_name": info.id.split("/")[1],
            "organization": info.author or info.id.split("/")[0],
            "hf_link": f"https://huggingface.co/{info.id}",
            "commit_hash": info.sha,
            "created_at": info.created_at.strftime("%Y-%m-%d %H:%M:%S UTC") if info.created_at else "Unknown",
            "last_modified": info.last_modified.strftime("%Y-%m-%d %H:%M:%S UTC") if info.last_modified else "Unknown",
            "downloads": info.downloads if info.downloads is not None else "Unknown",
            "likes": info.likes,
            "library_name": info.library_name or "Unknown",
            "pipeline_tag": info.pipeline_tag or "Unknown",
            "private": info.private,
            "gated": info.gated,
        }
        
        # Extract license information
        license_info = "Unknown"
        if info.card_data and isinstance(info.card_data, dict):
            license_info = info.card_data.get('license', 'Unknown')
        elif hasattr(info, 'tags') and info.tags:
            # Look for license in tags
            license_tags = [tag for tag in info.tags if tag.startswith('license:')]
            if license_tags:
                license_info = license_tags[0].replace('license:', '')
        model_data["license"] = license_info
        
        # Try to get paper link from card data or tags
        paper_link = "Not available"
        if info.card_data and isinstance(info.card_data, dict):
            card_str = str(info.card_data).lower()
            if 'arxiv' in card_str or 'paper' in card_str:
                paper_link = "Check model card for paper links"
        model_data["paper_link"] = paper_link
        
        # Extract config information if available
        if hasattr(info, 'config') and info.config:
            config = info.config
            
            # Extract architecture information
            architectures = config.get('architectures', [])
            model_data["architectures"] = architectures[0] if architectures else "Unknown"
            model_data["model_type"] = config.get('model_type', 'Unknown')
            
            # Look for quantization info
            quant_config = config.get('quantization_config', {})
            if quant_config:
                model_data["quantization"] = quant_config.get('quant_method', 'None')
            else:
                model_data["quantization"] = "None"
            
            # Try to extract model specifications from config
            # Check multiple possible field names for each specification
            
            # Hidden size (activation space)
            hidden_size = (config.get('hidden_size') or 
                          config.get('d_model') or 
                          config.get('n_embd') or 
                          config.get('embed_dim') or 
                          config.get('dim'))
            model_data["hidden_size"] = hidden_size if hidden_size is not None else "Unknown"
            
            # Number of layers
            num_layers = (config.get('num_hidden_layers') or 
                         config.get('n_layer') or 
                         config.get('num_layers') or 
                         config.get('n_layers') or
                         config.get('depth'))
            model_data["num_layers"] = num_layers if num_layers is not None else "Unknown"
            
            # Number of attention heads
            num_heads = (config.get('num_attention_heads') or 
                        config.get('n_head') or 
                        config.get('num_heads') or
                        config.get('attention_heads'))
            model_data["num_attention_heads"] = num_heads if num_heads is not None else "Unknown"
            
            # Vocabulary size
            vocab_size = config.get('vocab_size') or config.get('vocabulary_size') or config.get('n_vocab')
            model_data["vocab_size"] = vocab_size if vocab_size is not None else "Unknown"
            
            # Max position embeddings
            max_pos = (config.get('max_position_embeddings') or 
                      config.get('n_positions') or 
                      config.get('max_seq_len') or
                      config.get('max_sequence_length'))
            model_data["max_position_embeddings"] = max_pos if max_pos is not None else "Unknown"
            
            # Additional useful config info
            model_data["intermediate_size"] = config.get('intermediate_size', 'Unknown')
            model_data["num_key_value_heads"] = config.get('num_key_value_heads', 'Unknown')
            model_data["rope_scaling"] = config.get('rope_scaling', 'None')
            
            # Try to get tokenizer config info
            tokenizer_config = config.get('tokenizer_config', {})
            if tokenizer_config:
                model_data["bos_token"] = tokenizer_config.get('bos_token', 'Unknown')
                model_data["eos_token"] = tokenizer_config.get('eos_token', 'Unknown')
                model_data["pad_token"] = tokenizer_config.get('pad_token', 'Unknown')
                model_data["unk_token"] = tokenizer_config.get('unk_token', 'Unknown')
            else:
                model_data.update({
                    "bos_token": "Unknown",
                    "eos_token": "Unknown", 
                    "pad_token": "Unknown",
                    "unk_token": "Unknown"
                })
            
        else:
            # No config available
            model_data.update({
                "architectures": "Unknown",
                "model_type": "Unknown",
                "quantization": "Unknown",
                "hidden_size": "Unknown",
                "num_layers": "Unknown",
                "num_attention_heads": "Unknown",
                "vocab_size": "Unknown",
                "max_position_embeddings": "Unknown",
                "intermediate_size": "Unknown",
                "num_key_value_heads": "Unknown",
                "rope_scaling": "None",
                "bos_token": "Unknown",
                "eos_token": "Unknown", 
                "pad_token": "Unknown",
                "unk_token": "Unknown"
            })
        
        # Extract parameter count from safetensors info if available
        param_count = "Unknown"
        if hasattr(info, 'safetensors') and info.safetensors:
            if hasattr(info.safetensors, 'total'):
                # Convert to human readable format
                total_params = info.safetensors.total
                if total_params > 1e9:
                    param_count = f"{total_params/1e9:.1f}B"
                elif total_params > 1e6:
                    param_count = f"{total_params/1e6:.1f}M"
                else:
                    param_count = f"{total_params:,}"
        model_data["param_count"] = param_count
        
        # Extract additional useful information
        if hasattr(info, 'tags') and info.tags:
            model_data["tags"] = [tag for tag in info.tags if not tag.startswith('license:')][:10]  # Limit to first 10 non-license tags
        else:
            model_data["tags"] = []
            
        return model_data
        
    except (RepositoryNotFoundError, RevisionNotFoundError) as e:
        print(f"Warning: Could not retrieve model info for {hf_id}: {e}")
        return {
            "model_id": hf_id,
            "model_name": hf_id.split("/")[1],
            "organization": hf_id.split("/")[0],
            "hf_link": f"https://huggingface.co/{hf_id}",
            "commit_hash": "Unknown",
            "created_at": "Unknown",
            "last_modified": "Unknown",
            "downloads": "Unknown",
            "likes": "Unknown",
            "library_name": "Unknown",
            "pipeline_tag": "Unknown",
            "private": "Unknown",
            "gated": "Unknown",
            "license": "Unknown",
            "paper_link": "Not available",
            "architectures": "Unknown",
            "model_type": "Unknown",
            "quantization": "Unknown",
            "param_count": "Unknown",
            "hidden_size": "Unknown",
            "num_layers": "Unknown",
            "num_attention_heads": "Unknown",
            "vocab_size": "Unknown",
            "max_position_embeddings": "Unknown",
            "intermediate_size": "Unknown",
            "num_key_value_heads": "Unknown",
            "rope_scaling": "None",
            "bos_token": "Unknown",
            "eos_token": "Unknown", 
            "pad_token": "Unknown",
            "unk_token": "Unknown",
            "tags": []
        }

def create_model_readme(hf_id: str, model_path: str) -> bool:
    """Create a README.md file for the model directory."""
    try:
        model_data = get_model_info(hf_id)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        
        readme_content = f"""# {model_data['model_name']} Results

## Model Information

- **Model ID**: {model_data['model_id']}
- **Organization**: {model_data['organization']}
- **Model Name**: {model_data['model_name']}
- **Hugging Face Link**: [{model_data['model_id']}]({model_data['hf_link']})
- **Paper**: {model_data['paper_link']}
- **Model Version (Commit Hash)**: `{model_data['commit_hash']}`
- **Created**: {model_data['created_at']}
- **Last Modified**: {model_data['last_modified']}

## Model Specifications

### Architecture
- **Model Type**: {model_data['model_type']}
- **Architecture**: {model_data['architectures']}
- **Library**: {model_data['library_name']}
- **Pipeline Tag**: {model_data['pipeline_tag']}
- **License**: {model_data['license']}
- **Quantization**: {model_data['quantization']}

### Technical Details
- **Parameter Count**: {model_data['param_count']}
- **Number of Layers**: {model_data['num_layers']}
- **Number of Attention Heads**: {model_data['num_attention_heads']}
- **Hidden Size (Activation Space)**: {model_data['hidden_size']}
- **Vocabulary Size**: {model_data['vocab_size']}
- **Max Position Embeddings**: {model_data['max_position_embeddings']}"""

        # Add additional technical details if available
        if model_data.get('intermediate_size') != 'Unknown':
            readme_content += f"""
- **Intermediate Size**: {model_data['intermediate_size']}"""
        
        if model_data.get('num_key_value_heads') != 'Unknown':
            readme_content += f"""
- **Key-Value Heads**: {model_data['num_key_value_heads']}"""
        
        if model_data.get('rope_scaling') != 'None':
            readme_content += f"""
- **RoPE Scaling**: {model_data['rope_scaling']}"""

        readme_content += f"""

### Tokenizer Information
- **BOS Token**: {model_data['bos_token']}
- **EOS Token**: {model_data['eos_token']}
- **PAD Token**: {model_data['pad_token']}
- **UNK Token**: {model_data['unk_token']}

### Repository Details
- **Private**: {model_data['private']}
- **Gated**: {model_data['gated']}
- **Downloads**: {model_data['downloads']}
- **Likes**: {model_data['likes']}"""

        # Add tags if available
        if model_data['tags']:
            readme_content += f"""
- **Tags**: {', '.join(model_data['tags'])}"""

        readme_content += f"""

## Directory Structure

This directory contains experimental results for the {model_data['model_name']} model:

- `dataset/` - Generated prompt outputs from model inference
- `activations/` - Cached model activations for dataset generations
- `cautious_dir/` - Computed caution direction vectors from Chain-of-Thought analysis
- `attack_results/` - GCG-IRIS style prompt optimization results

## Notes

This README was auto-generated on model directory creation. Model specifications were extracted automatically from the Hugging Face Hub.

---
*Generated by [adv-steer](https://github.com/ky295/adv-steer) experiment framework on {timestamp}*
"""

        readme_path = os.path.join(model_path, "README.md")
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        print(f"Created README.md for {hf_id} at {readme_path}")
        return True
        
    except Exception as e:
        print(f"Error creating README for {hf_id}: {e}")
        return False

def make_dirs(hf_id: str) -> bool:
    validate_hf_id(hf_id)
    hf_path: str = os.path.join(hf_id.split("/")[0], hf_id.split("/")[1])
    model_path: str = os.path.join(top_level_dir, hf_path)

    # Create main model directory first
    if not os.path.exists(model_path):
        try:
            os.makedirs(model_path)
        except OSError as e:
            print(f"Error creating directory {model_path}: {e}")
            return False

    # Create subdirectories
    for subdir in subdirs:
        dir_path: str = os.path.join(model_path, subdir)
        if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path)
            except OSError as e:
                print(f"Error creating directory {dir_path}: {e}")
                return False
    
    # Create README.md for the model
    readme_path = os.path.join(model_path, "README.md")
    if not os.path.exists(readme_path):
        create_model_readme(hf_id, model_path)
    
    return True


def get_path(hf_id: str, subdir: str) -> str:
    hf_path = os.path.join(hf_id.split("/")[0], hf_id.split("/")[1])
    return os.path.join(top_level_dir, hf_path, subdir)
