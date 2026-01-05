"""
Perform weight orthogonalisation to create a model with the refusal direction ablated from its residual stream activations. Saves the refusal direction and orthogonalised model for each layer.

Supports both standard transformer architectures (DeepSeek, Qwen) and Mixture-of-Experts
architectures (gpt-oss). For MoE models, all expert down_proj weights are orthogonalized.

Usage:
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run -m interventions.create_ortho_model \
    --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --layers 16,17,18,19 \
    --type cot

For gpt-oss (MoE model):
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run -m interventions.create_ortho_model \
    --model_name openai/gpt-oss-20b \
    --layers 10,11,12,13 \
    --type cot
"""


import argparse
import gc
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def is_moe_model(model):
    """Check if model uses Mixture-of-Experts architecture."""
    if hasattr(model, 'model') and hasattr(model.model, 'layers') and len(model.model.layers) > 0:
        layer = model.model.layers[0]
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
            return True
    return False

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
    return parser.parse_args()

def load_model(model_name, device):
    """Load the model and tokenizer.

    Uses bfloat16 for gpt-oss models (mxfp4 quantization falls back to bf16),
    float16 for other CUDA models, and float32 for CPU.
    """
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Determine dtype based on model and device
    if device == "cuda" or device.startswith("cuda:"):
        # gpt-oss uses bfloat16 (mxfp4 falls back to bf16 without triton 3.4+)
        if "gpt-oss" in model_name:
            torch_dtype = torch.bfloat16
            print(f"Using bfloat16 for gpt-oss model")
        else:
            torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
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
    Memory-efficient implementation of orthogonalization for standard PyTorch format:
    W_out' = W_out - (vec·(vec^T·W_out))

    This avoids creating the full projection matrix by computing (vec^T·W_out) first.

    For standard nn.Linear: weight shape is [out_features, in_features]
    and computation is output = input @ weight.T

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


def get_orthogonalized_matrix_transposed(matrix, vec):
    """
    Memory-efficient implementation of orthogonalization for TRANSPOSED weight format.
    W' = W - (W @ vec) ⊗ vec^T

    This is for GPT-OSS MoE experts where:
    - Weight shape is [in_features, out_features] (transposed from standard)
    - Computation is output = input @ weight (no transpose!)
    - The output dimension is the LAST dimension

    Args:
        matrix: Weight matrix with shape [in_features, out_features]
        vec: Direction vector with shape [out_features] (matches last dimension)

    Returns:
        Orthogonalized matrix with shape [in_features, out_features]
    """
    # Ensure vector is the same dtype as matrix
    vec = vec.to(dtype=matrix.dtype)

    # The vector should match the LAST dimension (out_features) of the weight matrix
    if vec.size(0) != matrix.shape[1]:
        vec = resize_direction(vec, matrix.shape[1])

    # Step 1: Compute W @ vec (matrix-vector product)
    # This gives a vector of shape [in_features]
    matrix_vec = torch.matmul(matrix, vec)

    # Step 2: Compute (W @ vec) ⊗ vec^T without creating full projection matrix
    # This gives a matrix of shape [in_features, out_features]
    projection = torch.outer(matrix_vec, vec)

    # Step 3: Subtract the projection to get the orthogonalized matrix
    orthogonalized = matrix - projection

    # Free memory
    del matrix_vec, projection
    torch.cuda.empty_cache()

    return orthogonalized


def orthogonalize_moe_experts(experts, direction):
    """
    Orthogonalize MoE expert down_proj weights with respect to the given direction.

    For GPT-OSS MoE, experts use TRANSPOSED weight format:
    - down_proj has shape [num_experts, intermediate_size, hidden_size]
    - Computation is: output = input @ down_proj (no transpose!)
    - The output dimension (hidden_size) is the LAST dimension

    This requires the transposed orthogonalization formula:
    W' = W - (W @ d) ⊗ d^T

    Args:
        experts: The MoE experts module (e.g., GptOssExperts)
        direction: The direction to orthogonalize against, shape [hidden_size]
    """
    # Get the down_proj parameter - shape: [num_experts, intermediate_size, hidden_size]
    # Note: GPT-OSS uses transposed format where output dim is LAST
    down_proj = experts.down_proj.data
    num_experts = down_proj.shape[0]
    original_dtype = down_proj.dtype

    print(f"    Orthogonalizing {num_experts} expert down_proj weights (transposed format)...")
    print(f"    down_proj shape: {down_proj.shape} (experts, in_features, out_features)")
    print(f"    direction shape: {direction.shape} (should match out_features={down_proj.shape[2]})")

    # Ensure direction matches dtype
    direction = direction.to(dtype=down_proj.dtype, device=down_proj.device)

    # Orthogonalize each expert's down_proj
    for expert_idx in range(num_experts):
        # Extract single expert's weight: [intermediate_size, hidden_size]
        expert_weight = down_proj[expert_idx]

        # Apply TRANSPOSED orthogonalization (direction matches last dimension)
        orthogonalized = get_orthogonalized_matrix_transposed(expert_weight, direction)

        # Store back
        down_proj[expert_idx] = orthogonalized.to(dtype=original_dtype)

        # Periodic memory cleanup
        if (expert_idx + 1) % 8 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    # Update the parameter
    experts.down_proj.data = down_proj
    print(f"    Completed orthogonalization of {num_experts} experts")


def orthogonalize_model_weights(model, direction):
    """
    Orthogonalize key model weights with respect to the given direction
    using a memory-efficient approach.

    Supports both standard transformer architectures and Mixture-of-Experts (MoE).

    Args:
        model: The model to orthogonalize
        direction: The direction to orthogonalize against
    """
    print("Orthogonalizing model weights...")

    # Detect if this is an MoE model
    moe = is_moe_model(model)
    if moe:
        print("Detected Mixture-of-Experts architecture")

    # Ensure direction is on the right device
    direction = direction.to(device=model.device)

    # Process model weights in a memory-efficient manner
    gc.collect()
    torch.cuda.empty_cache()

    # Orthogonalize word embeddings
    # Embeddings have shape [vocab_size, hidden_size], and direction has shape [hidden_size].
    # Since direction matches the LAST dimension, we use the transposed formula.
    # This removes the refusal direction component from each embedding vector.
    print("Orthogonalizing word embeddings...")
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        model.model.embed_tokens.weight.data = get_orthogonalized_matrix_transposed(
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

        # MLP/MoE output projection
        if moe and hasattr(layer.mlp, 'experts'):
            # MoE layer - orthogonalize each expert's down_proj
            print("  Orthogonalizing MoE expert projections...")
            orthogonalize_moe_experts(layer.mlp.experts, direction)
            gc.collect()
            torch.cuda.empty_cache()
        elif hasattr(layer.mlp, 'down_proj'):
            # Standard MLP layer
            print("  Orthogonalizing MLP output projection...")
            layer.mlp.down_proj.weight.data = get_orthogonalized_matrix_efficient(
                layer.mlp.down_proj.weight.data, direction
            )

    print("Model weights orthogonalized successfully.")
    return model

def compute_refusal_dir(args, layer):
    """Compute refusal direction from cached activations.

    Uses bfloat16 for gpt-oss models, float16 for others.
    """
    # Determine dtype based on model
    if "gpt-oss" in args.model_name:
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    # Load activations
    print("Loading activations...")
    activations_refusal = load_activations(os.path.join('results', args.model_name, 'activations', 'refusal', f'layer_{layer}_{args.type}_activations.npy'))
    refusal_mean_act = get_mean_act(activations_refusal).to(dtype=dtype, device=args.device)
    # Free memory
    del activations_refusal
    gc.collect()
    torch.cuda.empty_cache()

    activations_nonrefusal = load_activations(os.path.join('results', args.model_name, 'activations', 'non_refusal', f'layer_{layer}_{args.type}_activations.npy'))
    nonrefusal_mean_act = get_mean_act(activations_nonrefusal).to(dtype=dtype, device=args.device)
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

    # Parse layers
    layers = [int(layer) for layer in args.layers.split(',')]

    for layer in layers:
        print(f"\n{'='*60}")
        print(f"Processing layer {layer}")
        print('='*60)

        # CRITICAL: Load fresh model for each layer to avoid cumulative orthogonalization
        # The orthogonalize_model_weights function modifies the model in-place,
        # so we must start from a fresh base model for each layer.
        print("Loading fresh base model...")
        model, tokenizer = load_model(args.model_name, args.device)

        print(f"Computing refusal direction for layer {layer}")
        refusal_dir = compute_refusal_dir(args, layer)

        # Save the tensor to a .pt file
        save_path = os.path.join('results', args.model_name, 'refusal_dir')
        os.makedirs(save_path, exist_ok=True)
        torch.save(refusal_dir, os.path.join(save_path, f'refusal_dir_{args.type}_layer_{layer}.pt'))
        print(f"refusal direction shape: {refusal_dir.shape}")

        # Orthogonalize model weights with respect to the refusal direction
        orthogonalized_model = orthogonalize_model_weights(model, refusal_dir)

        # Define the output directory
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

        # Free memory - delete model to prepare for next iteration
        del model, orthogonalized_model
        gc.collect()
        torch.cuda.empty_cache()

def run():
    main()

if __name__ == "__main__":
    main()