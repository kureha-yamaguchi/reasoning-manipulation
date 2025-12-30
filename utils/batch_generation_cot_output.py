"""
Script to generate n rollouts of CoT determined by --cot_repetitions and k rollouts of outputs (after </think>) determined by --output_repetitions. The script is split into 2 stages. Stage 1: Generate n responses for each prompt. Stage 2: For each CoT response, generate k different outputs (after the </think> tag). Invalid responses where the </think> tag is missing, is excluded from stage 2.

====================
Clean model paradigm
====================

Use to generate model outputs from the clean model using vllm with the training/ testing dataset train_harmful_prompts.csv or test_harmful_prompts.csv. Generations are saved in results/{model_name}/dataset/.

Example usage (for training dataset):
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run -m utils.batch_generation_cot_output \
  --input_csv train_harmful_prompts.csv \
  --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B

====================
Ortho model paradigm
====================

Use to generate model outputs from the locally stored orthogonalised model using vllm with evaluation dataset subset_5_test_harmful_prompts.csv or test_harmful_prompts.csv. Generations are saved in results/{model_name}/attack_results/.

Example usage (for ortho model created from activation at multiple layers, random subset of holdout dataset):
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run -m utils.batch_generation_cot_output \
  --input_csv subset_5_test_harmful_prompts.csv \
  --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --type cot \
  --layer 16,17,18,19

Example usage (for ortho model created from activation at a chosen single layer, full holdout dataset):
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run -m utils.batch_generation_cot_output \
  --input_csv test_harmful_prompts.csv \
  --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --type cot \
  --layer 17
"""

import argparse
import gc
from typing import List, Dict, Tuple, Union
import os
import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import questionary
import re
import torch
from huggingface_hub import hf_hub_download

from utils.paths import get_path, make_dirs

# Global variable for harmony format detection
HARMONY = False

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Two-stage CoT generation: Generate CoT responses, then generate outputs for each CoT"
    )
    parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", 
                        help="Model to use for generation")
    parser.add_argument("--input_csv", type=str, required=True,
                        help="CSV file to evaluate on")
    parser.add_argument("--type", type=str, default=None,
                        help="Ortho model from direction extracted from CoT tokens (cot) or 3 tokens at the end of prompt (baseline) or whole prompt (prompt)")
    parser.add_argument("--layer", type=str, default=None, 
                        help="Layer to take the activations and generate outputs from")
    parser.add_argument("--cot_repetitions", type=int, default=5, 
                        help="Number of CoT variations per prompt")
    parser.add_argument("--output_repetitions", type=int, default=5, 
                        help="Number of output variations per CoT")
    parser.add_argument("--max_new_tokens", type=int, default=2048, 
                        help="Maximum tokens for generation")
    parser.add_argument("--temperature", type=float, default=0.6, 
                        help="Temperature for sampling")
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="Batch size for vLLM inference")
    parser.add_argument("--tensor_parallel_size", type=int, default=None, 
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, 
                        help="GPU memory utilization ratio")
    parser.add_argument("--save_intermediate", action="store_true",
                        help="Save intermediate CoT results to separate file")
    parser.add_argument("--harmless", action="store_true", 
                        help="For harmless nonrefusal, harmful refusal   dataset configuration")
    return parser.parse_args()

def read_csv(input_csv: str) -> Union[List[str], None]:
    """Read prompts from CSV file."""
    print(f"Reading prompts from {input_csv}...")
    try:
        df = pd.read_csv(input_csv)
        prompts = df['prompt'].tolist()
        print(f"Loaded {len(prompts)} prompts")
        return prompts
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None

def save_csv(results: List[dict], output_csv: str):
    """Save results to CSV."""
    print(f"\nSaving {len(results)} results to {output_csv}...")
    try:
        output_df = pd.DataFrame(results)
        output_df.to_csv(output_csv, index=False)
        print(f"Results saved successfully to {output_csv}")
    except Exception as e:
        print(f"Error saving CSV: {e}")

def apply_chat_template_batch(prompts: List[str], model_name, tokenizer) -> List[str]:
    """Apply chat template to batch of prompts."""
    formatted_prompts = []

    if model_name == "mistralai/Magistral-Small-2506":
        SYSTEM_PROMPT = load_system_prompt(model_name, "SYSTEM_PROMPT.txt")
        for prompt in prompts:
            chat = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
            formatted_prompts.append(tokenizer.apply_chat_template(
                chat, add_generation_prompt=True, tokenize=False
            ))
    elif model_name == "nvidia/NVIDIA-Nemotron-Nano-9B-v2":
        for prompt in prompts:
            chat = [
                {"role": "system", "content": "/think"},
                {"role": "user", "content": prompt}
            ]
            formatted_prompts.append(tokenizer.apply_chat_template(
                chat, add_generation_prompt=True, tokenize=False
            ))
    else:
        for prompt in prompts:
            chat = [{"role": "user", "content": prompt}]
            formatted_prompts.append(tokenizer.apply_chat_template(
                chat, add_generation_prompt=True, tokenize=False
            ))

    
    return formatted_prompts

def load_system_prompt(repo_id: str, filename: str) -> str:
    file_path = hf_hub_download(repo_id=repo_id, filename=filename)
    with open(file_path, "r") as file:
        system_prompt = file.read()
    return system_prompt

def extract_cot_and_output(response):
    """
    Extract reasoning (CoT) and output portions from LLM responses.
    Supports both standard <think></think> tags and OpenAI harmony format.
    
    Args:
        response (str): The raw LLM response
        
    Returns:
        tuple: (cot_part, output_part, has_cot)
            - cot_part: The reasoning/thinking portion (or None if not found)
            - output_part: The final output portion
            - has_cot: Boolean indicating if CoT was found
    """

    global HARMONY
    
    if HARMONY:
        harmony_analysis_pattern = r'<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>'
        harmony_final_pattern = r'<\|start\|>assistant<\|channel\|>final<\|message\|>(.*?)<\|return\|>'
        
        analysis_match = re.search(harmony_analysis_pattern, response, re.DOTALL)
        final_match = re.search(harmony_final_pattern, response, re.DOTALL)
    
        if not analysis_match or not final_match:
            # Incomplete harmony format, treat as no full CoT
            return response, "", False
        cot_part = analysis_match.group(1)
        output_part = final_match.group(1)
        return cot_part, output_part, True
    
    # If harmony format not found, try standard <think></think> format
    # think_pattern = r'(.*?)(?=</think>)(.*)' # postive lookahead to avoid including </think> in cot_part
    think_pattern = r'(.*?</think>)(.*)'  # This captures </think> as part of group 1
    match = re.search(think_pattern, response, re.DOTALL)
    
    if match:
        cot_part = match.group(1)
        output_part = match.group(2)

        return cot_part, output_part, True
    
    # No CoT format found, return entire response as output
    return response, "", False



def stage1_generate_cots(
    llm: LLM, 
    tokenizer, 
    prompts: List[str], 
    args,
    sampling_params: SamplingParams
) -> Tuple[List[Dict], List[Dict]]:
    """
    Stage 1: Generate CoT responses for each prompt.
    Returns (valid_cot_results, invalid_cot_results)
    """
    print("\n=== STAGE 1: Generating CoT Responses ===")
    print(f"Generating {args.cot_repetitions} CoT variations for {len(prompts)} prompts")
    print(f"Total CoT generations: {len(prompts) * args.cot_repetitions}")
    
    valid_cot_results = []
    invalid_cot_results = []
    
    for batch_start in range(0, len(prompts), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(prompts))
        batch_prompts = prompts[batch_start:batch_end]
        
        print(f"\nProcessing batch {batch_start//args.batch_size + 1}: prompts {batch_start+1}-{batch_end}")
        
        # Create repeated prompts for this batch
        repeated_prompts = []
        prompt_indices = []
        
        for i, prompt in enumerate(batch_prompts):
            for rep in range(args.cot_repetitions):
                repeated_prompts.append(prompt)
                prompt_indices.append(i)
        
        # Apply chat template
        formatted_prompts = apply_chat_template_batch(repeated_prompts, args.model_name, tokenizer)
        
        # Generate
        outputs = llm.generate(formatted_prompts, sampling_params)
        
        # Process outputs
        for i, output in enumerate(outputs):
            original_idx = prompt_indices[i]
            original_prompt = batch_prompts[original_idx]

            # full_response = output.outputs[0].text
            full_response = tokenizer.decode(output.outputs[0].token_ids, skip_special_tokens=False)

            cot_rep = (i % args.cot_repetitions) + 1
            
            # Extract CoT and output parts
            cot_part, output_part, has_valid_cot = extract_cot_and_output(full_response)
            
            # cot_part = getattr(output.outputs[0], 'reasoning_content', None)
            # output_part = getattr(output.outputs[0], 'content', None) or full_response
            # has_valid_cot = cot_part is not None and output_part.strip() != ""


            result = {
                "prompt": original_prompt,
                "full_cot_response": full_response,
                "cot_part": cot_part,
                "output_part": output_part,
                "cot_repetition": cot_rep,
                "has_valid_cot": has_valid_cot
            }
            
            if has_valid_cot:
                valid_cot_results.append(result)
            else:
                invalid_cot_results.append(result)
        
        gc.collect()
    
    print(f"\nStage 1 complete:")
    print(f"  - Valid CoT responses (with </think>): {len(valid_cot_results)}")
    print(f"  - Invalid CoT responses (missing </think>): {len(invalid_cot_results)}")
    
    # Show breakdown by prompt if there are invalid responses
    if invalid_cot_results:
        prompt_stats = {}
        for result in valid_cot_results:
            prompt = result["prompt"][:50] + "..." if len(result["prompt"]) > 50 else result["prompt"]
            prompt_stats[prompt] = prompt_stats.get(prompt, 0) + 1
        
        print("\n  Valid CoT counts per prompt:")
        for prompt, count in prompt_stats.items():
            print(f"    - {prompt}: {count}/{args.cot_repetitions}")
    
    return (valid_cot_results, invalid_cot_results)



def extract_final_message(text):
    """
    Extract the final message content from model output.
    
    Args:
        text (str): The model's generated output
        
    Returns:
        str: The clean final message content
    """
    # Single regex to capture everything up to any special token or end of string
    pattern = r'(.*?)(?:<\|return\|>|<｜end▁of▁sentence｜>|$)'
    
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1)
    

def stage2_generate_outputs(
    llm: LLM,
    tokenizer,
    cot_results: List[Dict],
    args,
    sampling_params: SamplingParams
) -> List[Dict]:
    """
    Stage 2: For each CoT response, generate multiple output variations.
    Optimized version with batch template processing.
    """

    global HARMONY

    print("\n=== STAGE 2: Generating Output Variations ===")
    print(f"Generating {args.output_repetitions} outputs for {len(cot_results)} CoT responses")
    print(f"Total output generations: {len(cot_results) * args.output_repetitions}")
    
    all_final_results = []
    
    for batch_start in range(0, len(cot_results), args.batch_size // args.output_repetitions):
        batch_end = min(batch_start + args.batch_size // args.output_repetitions, len(cot_results))
        batch_cots = cot_results[batch_start:batch_end]
        
        print(f"\nProcessing batch: CoT responses {batch_start+1}-{batch_end}")
        
        # First, collect all unique prompts for batch template application
        unique_prompts = [cot_result["prompt"] for cot_result in batch_cots]
        
        # Apply chat template to all prompts in batch
        formatted_prompts = apply_chat_template_batch(unique_prompts, args.model_name, tokenizer)
        
        # Create prompt-cot sequences for generation
        rep_combined_input = []
        cot_indices = []
        
        for i, (cot_result, formatted_prompt) in enumerate(zip(batch_cots, formatted_prompts)):
            # Concatenate the formatted prompt with the CoT part
            if HARMONY:
                combined_input = formatted_prompt + "<|channel|>analysis<|message|>" + cot_result["cot_part"] + "<|end|><|start|>assistant<|channel|>final<|message|>"
            else:
                combined_input = formatted_prompt + cot_result["cot_part"]
                # combined_input = formatted_prompt + cot_result["cot_part"] + "\n</think>"
            # Create multiple copies for output repetitions
            for rep in range(args.output_repetitions):
                rep_combined_input.append(combined_input)
                cot_indices.append(i)
        
        # Generate outputs
        outputs = llm.generate(rep_combined_input, sampling_params)
        
        # Process outputs
        for i, output in enumerate(outputs):
            cot_idx = cot_indices[i]
            cot_result = batch_cots[cot_idx]
            output_rep = (i % args.output_repetitions) + 1
            
            # Extract the generated output
            # generated_text = output.outputs[0].text
            generated_text = tokenizer.decode(output.outputs[0].token_ids, skip_special_tokens=False)

            final_response = extract_final_message(generated_text)
        
            all_final_results.append({
                "prompt": cot_result["prompt"],
                "cot": cot_result["cot_part"],
                "output": final_response,
                "cot_rep_n": cot_result["cot_repetition"],
                "output_rep_n": output_rep
            })
        
        gc.collect()
    
    print(f"\nStage 2 complete: Generated {len(all_final_results)} final outputs")
    return all_final_results

def generate_and_save(llm: LLM, tokenizer, input_csv: str, output_csv: str, prompts, sampling_params, args) -> None:
    # Stage 1: Generate CoTs
    valid_cot_results, invalid_cot_results = stage1_generate_cots(
        llm, tokenizer, prompts, args, sampling_params
    )
    gc.collect()
    torch.cuda.empty_cache()  # Clear CUDA memory too
    
    # Save intermediate results if requested
    if args.save_intermediate:
        # Save valid CoT results
        if valid_cot_results:
            valid_cot_csv = input_csv.replace('.csv', f'_valid_cot_{args.cot_repetitions}.csv')
            valid_cot_path = os.path.join(get_path(args.model_name, 'dataset'), os.path.basename(valid_cot_csv))
            save_csv(valid_cot_results, valid_cot_path)
        
        # Save invalid CoT results for debugging
        if invalid_cot_results:
            invalid_cot_csv = input_csv.replace('.csv', f'_invalid_cot_{args.cot_repetitions}.csv')
            save_csv(invalid_cot_results, invalid_cot_csv)
            print(f"  Saved {len(invalid_cot_results)} invalid CoT responses for review")
    
    # # Check if we have valid CoT results to proceed
    # if not valid_cot_results:
    #     print(f"\n No valid CoT responses generated (all missing </think> tag)")
    #     print(f"   Cannot proceed to Stage 2")
    #     return
    
    # Stage 2: Generate outputs for each valid CoT
    final_results = stage2_generate_outputs(
        llm, tokenizer, valid_cot_results, args, sampling_params
    )
    
    # Save final results
    save_csv(final_results, output_csv)
    
    print(f"\n✓ Completed {input_csv}")
    print(f"  - Original prompts: {len(prompts)}")
    print(f"  - Valid CoT responses: {len(valid_cot_results)}")
    print(f"  - Invalid CoT responses: {len(invalid_cot_results)}")
    print(f"  - Final outputs: {len(final_results)}")

def main():
    args = parse_args()

    if not args.tensor_parallel_size:
        # if not explicitly specified, use all available CUDA devices
        args.tensor_parallel_size = torch.cuda.device_count() 

    global HARMONY
    HARMONY = "gpt-oss" in args.model_name

    print(f"CUDA available: {torch.cuda.is_available()}")
    gc.collect()
    torch.cuda.empty_cache()

     # Handle CSV file selection
    input_csv = os.path.join('dataset', args.input_csv)

    # Read prompts
    prompts = read_csv(input_csv)
    
    # Set up sampling parameters
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    # Ortho model paradigm
    if args.type is not None and args.layer is not None:
        # Process each layer (if there are multiple layers)
        for layer in args.layer.split(','):
            layer = layer.strip()  # Remove any whitespace
            # Construct local model path
            if args.harmless:
                local_model_path = os.path.join('results', args.model_name, f'ortho_model_{args.type}_layer_{layer}_harmless')
            else:
                local_model_path = os.path.join('results', args.model_name, f'ortho_model_{args.type}_layer_{layer}')
            print(f"Loading model from local path: {local_model_path}")

            # Initialize model and tokenizer
            print("Loading tokenizer...")
            
            if args.model_name == "mistralai/Magistral-Small-2506":
                # Use base model tokenizer as workaround
                tokenizer = AutoTokenizer.from_pretrained(
                    "unsloth/Magistral-Small-2506", 
                    trust_remote_code=True
                )
                print("Using base model tokenizer for Magistral-Small-2506")
            else:
                tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
            
            print("Initializing vLLM...")
            
            # Special vLLM configuration for Magistral models
            if args.model_name == "mistralai/Magistral-Small-2506":
                # Set environment variable for context length
                os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
                llm = LLM(
                    model=local_model_path,
                    tensor_parallel_size=args.tensor_parallel_size,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    trust_remote_code=True,
                    tokenizer_mode="mistral",  # Use mistral tokenizer mode
                    config_format="mistral",   # Use mistral config format
                    load_format="mistral",     # Use mistral load format
                )

            else:
                llm = LLM( 
                    model=local_model_path,
                    tensor_parallel_size=args.tensor_parallel_size,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    trust_remote_code=True,
                )
                
            print("Model loaded successfully!")

            # Construct output CSV name
            input_csv_name = os.path.splitext(args.input_csv)[0]
            if args.harmless:
                output_csv = os.path.join('results', args.model_name, 'attack_results', f'ortho_output_{input_csv_name}_{args.type}_layer_{layer}_harmless.csv')
            else:
                output_csv = os.path.join('results', args.model_name, 'attack_results', f'ortho_output_{input_csv_name}_{args.type}_layer_{layer}.csv')
            
            generate_and_save(llm, tokenizer, input_csv, output_csv, prompts, sampling_params, args)

            del llm, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            
            print(f"\n{'='*60}")
            print(f"✅ Processed layer {layer}")
            print(f"{'='*60}")

    # Clean model paradigm
    else:
        # Initialize model and tokenizer
        print("Loading tokenizer...")
        
        if args.model_name == "mistralai/Magistral-Small-2506":
            # Use base model tokenizer as workaround
            tokenizer = AutoTokenizer.from_pretrained(
                "unsloth/Magistral-Small-2506", 
                trust_remote_code=True
            )
            print("Using base model tokenizer for Magistral-Small-2506")
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        
        print("Initializing vLLM...")
        
        # Special vLLM configuration for Magistral models
        if args.model_name == "mistralai/Magistral-Small-2506":
            # Set environment variable for context length
            os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
            llm = LLM(
                model=args.model_name,
                tensor_parallel_size=args.tensor_parallel_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
                trust_remote_code=True,
                tokenizer_mode="mistral",  # Use mistral tokenizer mode
                config_format="mistral",   # Use mistral config format
                load_format="mistral",     # Use mistral load format
            )

        else:
            llm = LLM(
                model=args.model_name,
                tensor_parallel_size=args.tensor_parallel_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
                trust_remote_code=True,
            )
            
        print("Model loaded successfully!")

        # Construct output CSV name
        input_csv_name = os.path.splitext(args.input_csv)[0]
        if args.harmless:
            output_csv = os.path.join('results', args.model_name, 'dataset', f'{input_csv_name}_cot{args.cot_repetitions}_out{args.output_repetitions}_harmless.csv')
        else:
            output_csv = os.path.join('results', args.model_name, 'dataset', f'{input_csv_name}_cot{args.cot_repetitions}_out{args.output_repetitions}.csv')
        
        generate_and_save(llm, tokenizer, input_csv, output_csv, prompts, sampling_params, args)

        del llm, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        
        print(f"\n{'='*60}")
        print(f"✅ Processing complete")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()