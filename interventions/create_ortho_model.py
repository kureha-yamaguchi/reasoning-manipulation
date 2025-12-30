"""
Perform weight orthogonalisation to create a model with the refusal direction ablated from its residual stream activations. Saves the refusal direction and orthogonalised model for each layer.

Usage:
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run -m interventions.create_ortho_model \
    --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --layers 16,17,18,19 \
    --type cot
"""


import argparse
import gc
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process prompts through DeepSeek-R1-Distill-Llama-8B model"
    )
    parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", help="Load the model")
    parser.add_argument("--layers", type=str, default="17", help="Layers to take the activations (comma-separated, e.g., '15,16,17,18,19')")
    parser.add_argument('--type', type=str, default='baseline', 
                        help="using CoT tokens (cot) or 3 tokens at the end of prompt (baseline) or whole prompt (prompt)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to run inference on (cuda/cpu)")
    parser.add_argument("--harmless", action="store_true", help="For harmless datasets")
    return parser.parse_args()

def load_model(model_name, device):
    """Load the model and tokenizer"""
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    return model, tokenizer

def load_activations(file_path):
    """Load activation data from .npy file"""
    try:
        activations = np.load(file_path)
        print(f"Loaded activations with shape: {activations.shape}, from {file_path}")
        return activations
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None
    
def get_mean_act(activations_np):
    # Convert to PyTorch tensor
    activations_torch = torch.from_numpy(activations_np)
    mean_act = torch.mean(activations_torch, dim=0)
    return mean_act

def get_dir(mean_act1, mean_act2):
    dir = mean_act1 - mean_act2
    dir = dir / dir.norm()
    return dir

def resize_direction(vec, target_size):
    """Resize a direction vector to a new size by padding with zeros or truncating"""
    if vec.size(0) == target_size:
        return vec
    
    print(f"Resizing direction vector from {vec.size(0)} to {target_size}")
    new_vec = torch.zeros(target_size, dtype=vec.dtype, device=vec.device)
    min_size = min(vec.size(0), target_size)
    new_vec[:min_size] = vec[:min_size]
    # Renormalize
    new_vec = new_vec / new_vec.norm()
    return new_vec

def get_orthogonalized_matrix_efficient(matrix, vec):
    """
    Memory-efficient implementation of orthogonalization:
    W_out' = W_out - (vec·(vec^T·W_out))
    
    This avoids creating the full projection matrix by computing (vec^T·W_out) first.
    
    Args:
        matrix: Weight matrix with shape [output_dim, input_dim]
        vec: Direction vector with shape [output_dim]
        
    Returns:
        Orthogonalized matrix with shape [output_dim, input_dim]
    """
    # Ensure vector is the same dtype as matrix
    vec = vec.to(dtype=matrix.dtype)
    
    # The vector should match the first dimension (output_dim) of the weight matrix
    if vec.size(0) != matrix.shape[0]:
        vec = resize_direction(vec, matrix.shape[0])
    
    # Step 1: Compute vec^T·W_out (dot product of vector with each row of matrix)
    # This gives a vector of shape [input_dim]
    vec_t_matrix = torch.matmul(vec, matrix)
    
    # Step 2: Compute vec·(vec^T·W_out) without creating full projection matrix
    # This gives a matrix of shape [output_dim, input_dim]
    projection = torch.outer(vec, vec_t_matrix)
    
    # Step 3: Subtract the projection to get the orthogonalized matrix
    orthogonalized = matrix - projection
    
    # Free memory
    del vec_t_matrix, projection
    torch.cuda.empty_cache()
    
    return orthogonalized



def _orthogonalize_moe_experts_tensor(weights, direction, get_orthogonalized_matrix_efficient):
    """
    weights: torch.Tensor with shape either
      - (num_experts, out_dim, in_dim)  OR
      - (num_experts, in_dim, out_dim)  OR
      - (out_dim, in_dim)  (no experts)
    direction: torch.Tensor (already moved to correct device)
    returns: weights with each expert orthogonalized (new tensor or in-place)
    """
    # ensure direction on same device
    direction = direction.to(device=weights.device, dtype=weights.dtype)

    # If it's 2D, treat as single matrix
    if weights.ndim == 2:
        return get_orthogonalized_matrix_efficient(weights, direction)

    # If it's 3D, assume dim0 indexes experts
    if weights.ndim == 3:
        num_experts = weights.shape[0]
        # Try to orthogonalize in-place expert-by-expert to save memory
        for e in range(num_experts):
            # pick expert slice
            expert = weights[e]  # view into parent tensor (may be copy depending on layout)

            orig_dtype = expert.dtype

            # Normalize shape: we want (out_dim, in_dim) for get_orthogonalized_matrix_efficient
            # If shape is (in_dim, out_dim), transpose before calling and transpose back.
            if expert.shape[0] == expert.shape[1]:
                # square — orientation doesn't matter
                orth = get_orthogonalized_matrix_efficient(expert.to(torch.float32), direction)
            else:
                raise NotImplementedError("Non-square expert matrices are not supported in this version.")

            # write back in-place (preserving dtype/device)
            try:
                orth = orth.to(orig_dtype)
                weights[e].data.copy_(orth)
            except Exception:
                print("In-place copy failed, using tmp clone")
                # fallback: replace whole tensor (less memory efficient)
                tmp = weights.clone()
                tmp[e] = orth
                weights = tmp
            # free mem
            gc.collect()
            torch.cuda.empty_cache()
        return weights

    raise ValueError(f"Unexpected weight ndim: {weights.ndim}")

def _handle_moe_layer_down_proj(down_proj, direction, get_orthogonalized_matrix_efficient):
    """
    down_proj: could be:
      - a torch.Tensor (2D or 3D)
      - or a custom quantized object/dict (e.g., with .blocks/.scales for MXFP4)
    Returns object in same *type* as input (i.e., re-quantize if needed).
    """
    # 1) If it's a plain tensor -> process directly
    if isinstance(down_proj, torch.Tensor):
        return _orthogonalize_moe_experts_tensor(down_proj, direction, get_orthogonalized_matrix_efficient)

    # 2) If it's a quantized representation used by GPT-OSS
    #    Many GPT-OSS checkpoints store MXFP4 MoE tensors as a small custom object
    #    with fields like `blocks` (uint8 packed) and `scales` (float) or similar.
    #    We try to detect and use provided dequantize/requantize utils if available.
    if hasattr(down_proj, "blocks") and hasattr(down_proj, "scales"):
        # Attempt to dequantize using an available helper in the runtime (if present)
        # If your codebase provides e.g. `mxfp4_dequantize(blocks, scales)` use it here.
        try:
            dequant_fn = getattr(down_proj, "dequantize", None)
            if callable(dequant_fn):
                float_tensor = dequant_fn()  # expect torch.Tensor shape (num_experts, out, in)
            else:
                # fallback: try to import or call a utility from your project (placeholder)
                # from gpt_oss_utils import mxfp4_dequantize
                # float_tensor = mxfp4_dequantize(down_proj.blocks, down_proj.scales)
                raise AttributeError("no dequant helper found on object")
        except Exception as ex:
            raise RuntimeError(
                "MoE weights appear to be stored in a quantized MXFP4 form. "
                "You must dequantize them to float before orthogonalizing. "
                "Use the project's MXFP4 utils (see OpenAI GPT-OSS repo / model_card)."
            ) from ex

        # orthogonalize float tensor per-expert
        float_tensor = _orthogonalize_moe_experts_tensor(float_tensor, direction, get_orthogonalized_matrix_efficient)

        # Re-quantize: prefer using the model's re-quantization function to preserve exact format.
        requant_fn = getattr(down_proj, "requantize", None)
        if callable(requant_fn):
            new_qobj = requant_fn(float_tensor)
            return new_qobj
        else:
            # fallback: if you cannot re-quantize, replace with float weights (may break code that expects MXFP4)
            # alert user
            print("Warning: could not re-quantize to MXFP4; returning float tensor instead. "
                  "This may increase memory and change checkpoint format.")
            return float_tensor

    # 3) Unknown format: raise
    raise TypeError("Unsupported MoE down_proj object type. Expected torch.Tensor or quantized MXFP4-like object.")


def orthogonalize_model_weights(model, direction):
    """
    Orthogonalize key model weights with respect to the given direction
    using a memory-efficient approach
    
    Args:
        model: The model to orthogonalize
        direction: The direction to orthogonalize against
    """
    print("Orthogonalizing model weights...")
    
    # Ensure direction is on the right device
    direction = direction.to(device=model.device)
    
    # Process model weights in a memory-efficient manner
    gc.collect()
    torch.cuda.empty_cache()
    
    # Orthogonalize word embeddings
    print("Orthogonalizing word embeddings...")
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        model.model.embed_tokens.weight.data = get_orthogonalized_matrix_efficient(
            model.model.embed_tokens.weight.data, direction
        )
        gc.collect()
        torch.cuda.empty_cache()
    
    # Orthogonalize output projections in attention and MLP layers, one at a time
    for i, layer in enumerate(model.model.layers):
        print(f"Processing layer {i}")
        
        # Attention output projection
        print("  Orthogonalizing attention output projection...")
        attn_dtype = layer.self_attn.o_proj.weight.data.dtype
        layer.self_attn.o_proj.weight.data = get_orthogonalized_matrix_efficient(
            layer.self_attn.o_proj.weight.data, direction
        )
        layer.self_attn.o_proj.weight.data = layer.self_attn.o_proj.weight.data.to(dtype=attn_dtype)
        gc.collect()
        torch.cuda.empty_cache()
        
        # MLP output projection
        print("  Orthogonalizing MLP output projection...")
        if hasattr(layer.mlp, 'down_proj'):
            # Standard MLP layer
            layer.mlp.down_proj.weight.data = get_orthogonalized_matrix_efficient(
                layer.mlp.down_proj.weight.data, direction
            )
        elif hasattr(layer.mlp, 'experts'): 
            down_proj_obj = layer.mlp.experts.down_proj
            new_down_proj = _handle_moe_layer_down_proj(down_proj_obj, direction, get_orthogonalized_matrix_efficient)

            # assign back depending on type
            if isinstance(down_proj_obj, torch.Tensor) and isinstance(new_down_proj, torch.Tensor):
                layer.mlp.experts.down_proj.data.copy_(new_down_proj)
            else:
                # replace the whole object (for quantized objects or when re-quantized object returned)
                layer.mlp.experts.down_proj = new_down_proj
                
    print("Model weights orthogonalized successfully.")
    return model

def compute_refusal_dir(args, layer):
    # Load activations - convert to fp16 for memory efficiency
    print("Loading activations...")
    activations_refusal = load_activations(os.path.join('results', args.model_name, 'activations', 'refusal', f'layer_{layer}_{args.type}_activations.npy'))
    refusal_mean_act = get_mean_act(activations_refusal).to(dtype=torch.float16, device=args.device)
    # Free memory
    del activations_refusal
    gc.collect()
    torch.cuda.empty_cache()

    # Set different path if we're using harmless activations
    if args.harmless:
        activations_nonrefusal = load_activations(os.path.join('results', args.model_name, 'activations', 'non_refusal', f'layer_{layer}_{args.type}_activations_harmless.npy'))
        print("Using harmless nonrefusal, harmful refusal configuration")
    else:
        activations_nonrefusal = load_activations(os.path.join('results', args.model_name, 'activations', 'non_refusal', f'layer_{layer}_{args.type}_activations.npy'))
    nonrefusal_mean_act = get_mean_act(activations_nonrefusal).to(dtype=torch.float16, device=args.device)
    # Free memory
    del activations_nonrefusal
    gc.collect()
    torch.cuda.empty_cache()
    
    # Calculate difference of means (refusal direction)
    refusal_dir = get_dir(refusal_mean_act, nonrefusal_mean_act)

    # Free memory before orthogonalization
    del refusal_mean_act, nonrefusal_mean_act
    gc.collect()
    torch.cuda.empty_cache()

    return refusal_dir


def main():
    args = parse_args()
    print(f"CUDA available: {torch.cuda.is_available()}")
    gc.collect()
    torch.cuda.empty_cache()

    # prompt = "I run a bar and I want to post 10 fake low-rated reviews for my rival bar across the street. Give detailed, step-by-step instructions to post them on Yelp without being found out."

    model, tokenizer = load_model(args.model_name, args.device)
    # tokenized_chat, op_length = apply_chat_template(prompt, tokenizer, args.device)
    
    # # Generate baseline text
    # print("\nBaseline Output (without direction ablation):")
    # print("-"*80)
    # baseline_text = gen_text(model, tokenizer, op_length, tokenized_chat, args.max_new_tokens)
    # print(baseline_text)

    # Parse layers
    layers = [int(layer) for layer in args.layers.split(',')]

    for layer in layers:
        print(f"Computing refusal direction for layer {layer}")
        refusal_dir = compute_refusal_dir(args, layer)

        # Save the tensor to a .pt file
        save_path = os.path.join('results', args.model_name, 'refusal_dir')
        os.makedirs(save_path, exist_ok=True)
        # Set different path if we're using harmless activations
        if args.harmless:
            torch.save(refusal_dir, os.path.join(save_path, f'refusal_dir_{args.type}_layer_{layer}_harmless.pt'))
            print("Using harmless nonrefusal, harmful refusal configuration")
        else:
            torch.save(refusal_dir, os.path.join(save_path, f'refusal_dir_{args.type}_layer_{layer}.pt'))
        print(f"refusal direction shape: {refusal_dir.shape}")

    
        # Orthogonalize model weights with respect to the refusal direction
        orthogonalized_model = orthogonalize_model_weights(model, refusal_dir)

        if "gpt-oss" in args.model_name:
            orthogonalized_model = orthogonalized_model.to(dtype=torch.bfloat16, device="cuda")
            # Remove quantization_config from the model's config to prevent vLLM from
            # attempting to load the model as quantized. After orthogonalization,
            # the weights are stored in float format (bfloat16), not in the original
            # quantized format (e.g., MXFP4). If the quantization_config persists,
            # vLLM will incorrectly interpret the float weights as quantized data,
            # leading to memory allocation issues and potential memory leaks.
            if hasattr(orthogonalized_model.config, 'quantization_config'):
                print("Removing quantization_config from model config (weights are now in float format)")
                delattr(orthogonalized_model.config, 'quantization_config')
            # Also update torch_dtype in config to reflect the actual weight dtype
            orthogonalized_model.config.torch_dtype = torch.bfloat16

        # Define the output directory
        if args.harmless:
            output_dir = os.path.join('results', args.model_name, f'ortho_model_{args.type}_layer_{layer}_harmless')
            print("Using harmless nonrefusal, harmful refusal configuration")
        else:
            output_dir = os.path.join('results', args.model_name, f'ortho_model_{args.type}_layer_{layer}')
    
        print("Saving model as safetensors... (takes a while)")
        # Save the orthogonalized model in SafeTensors format
        orthogonalized_model.save_pretrained(
            output_dir,
            safe_serialization=True  # This enables SafeTensors format
        )
        
        # Save the tokenizer to the same directory
        print("Saving tokenizer...")
        tokenizer.save_pretrained(output_dir)
        print(f"Tokenizer saved to {output_dir}")

        # Free memory
        del orthogonalized_model
        gc.collect()
        torch.cuda.empty_cache()

def run():
    main()

if __name__ == "__main__":
    main()