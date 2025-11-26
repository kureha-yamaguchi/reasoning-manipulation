import argparse
import gc
from typing import List, Union
import os
import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import questionary 
import torch

from utils.paths import get_path, make_dirs, validate_hf_id

# CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m utils.batch_csv_generation --input_dir dataset/ --input_csv all_harmful_prompts.csv --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B

# for prerelease versions to support openai-oss need to run with:
# CUDA_VISIBLE_DEVICES=2,3 uv run --index-strategy unsafe-best-match --prerelease=allow -m utils.batch_csv_generation --model_name=openai/gpt-oss-20b --tensor_parallel_size=2 

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process prompts through the model using vLLM with repetitions - supports multiple CSV files"
    )
    parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", 
                        help="Load the model")
    parser.add_argument("--input_dir", type=str, default="dataset/base/", 
                        help="Dataset input CSV directory")
    parser.add_argument('--input_csv', type=str, nargs='*',  
                        help='Filenames of the input CSV files with prompts, e.g. advbench_prompts.csv. Use "all" to process all CSV files in the input directory.')
    parser.add_argument("--max_new_tokens", type=int, default=2048, 
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.6, 
                        help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.95, 
                        help="Top-p value for nucleus sampling")
    parser.add_argument("--repetitions", type=int, default=5, 
                        help="Number of times to repeat generation for each prompt")
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="Batch size for vLLM inference")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, 
                        help="Number of GPUs to use for tensor parallelism")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, 
                        help="GPU memory utilization ratio")
    return parser.parse_args()

def read_csv(input_csv: str, dataset_dir:str) -> Union[List[str], None]:
    """Read prompts from the CSV file."""
    print(f"Reading prompts from {input_csv}...")
    try:
        df = pd.read_csv(os.path.join(dataset_dir, input_csv))
        prompts = df['prompt'].tolist()
        print(f"Loaded {len(prompts)} prompts from the CSV file")
        return prompts
    except Exception as e:
        print(f"Error reading input CSV: {e}")
        return None

def save_csv(results: List[dict], output_csv: str, dataset_dir: str):
    """Save results to CSV."""
    print(f"\nSaving results to {output_csv}...")
    try:
        output_df = pd.DataFrame(results)
        output_df.to_csv(os.path.join(dataset_dir, output_csv), index=False)
        print(f"Results saved successfully to {output_csv}")
        print(f"Total rows saved: {len(results)}")
    except Exception as e:
        print(f"Error saving output CSV: {e}")

def apply_chat_template_batch(prompts: List[str], tokenizer) -> List[str]:
    """Apply chat template to a batch of prompts."""
    formatted_prompts = []
    for prompt in prompts:
        chat = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            chat, 
            add_generation_prompt=True, 
            tokenize=False
        )
        formatted_prompts.append(formatted_prompt)
    return formatted_prompts

def process_batch(llm: LLM, tokenizer, prompts_batch: List[str], 
                 sampling_params: SamplingParams, repetitions: int) -> List[dict]:
    """Process a batch of prompts with repetitions."""
    results = []
    
    # Apply chat template to the prompts
    formatted_prompts = apply_chat_template_batch(prompts_batch, tokenizer)
    
    print(f"Generating responses for {len(formatted_prompts)} prompts (batch size: {len(prompts_batch)}")
    
    # Generate responses
    outputs = llm.generate(formatted_prompts, sampling_params)
    
    # Process outputs and group by original prompt
    for i, output in enumerate(outputs):
        original_prompt = prompts_batch
        response = output.outputs[0].text
        
        results.append({
            "prompt": original_prompt,
            "response": response,
        })
    
    return results

def process_single_csv(llm: LLM, tokenizer, input_csv: str, args, sampling_params: SamplingParams) -> None:
    """Process a single CSV file."""
    print(f"\n{'='*60}")
    print(f"Processing CSV file: {input_csv}")
    print(f"{'='*60}")
    
    # Read prompts
    prompts = read_csv(input_csv, args.input_dir)
    if prompts is None:
        print(f"Skipping {input_csv} due to read error.")
        return
    
    print(f"Processing {len(prompts)} prompts with {args.repetitions} repetitions each...")
    print(f"Total generations: {len(prompts) * args.repetitions}")
    
    all_results = []
    
    # Process prompts in batches
    for i in range(0, len(prompts), args.batch_size):
        batch_end = min(i + args.batch_size, len(prompts))
        prompts_batch = prompts[i:batch_end]
        
        print(f"\nProcessing batch {i//args.batch_size + 1}/{(len(prompts) + args.batch_size - 1)//args.batch_size}")
        print(f"Batch range: {i+1}-{batch_end} of {len(prompts)} prompts")
        
        batch_results = process_batch(llm, tokenizer, prompts_batch, sampling_params, args.repetitions)
        all_results.extend(batch_results)
        
        # Clean up memory
        gc.collect()
    
    # Save results
    output_csv = input_csv.replace('_prompts.csv', f'_outputs_{args.repetitions}.csv')
    save_csv(all_results, output_csv, get_path(args.model_name, 'dataset'))
    print(f"Completed processing {input_csv}! Generated {len(all_results)} total responses.")

def main():
    args = parse_args()
    print(f"CUDA available: {torch.cuda.is_available()}")
    gc.collect()
    torch.cuda.empty_cache()
    # Handle CSV file selection
    input_csvs = args.input_csv
    available_csvs = [f for f in os.listdir(args.input_dir) if f.endswith('.csv')]
    
    if not input_csvs:
        input_csvs = questionary.checkbox(
            "Select CSV files to process (none specified via --input_csv):",
            choices=available_csvs
        ).ask()
        
        if not input_csvs:
            print("No CSV files selected. Exiting.")
            return
    elif len(input_csvs) == 1 and input_csvs[0].lower() == 'all':
        input_csvs = available_csvs
        print(f"Processing ALL CSV files in {args.input_dir}: {', '.join(input_csvs)}")

    make_dirs(args.model_name)

    # Initialize tokenizer and model once for all files
    print("Loading tokenizer for chat template...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    print("Initializing vLLM...")
    llm = LLM(
        model=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        # max_model_len=4096,  # Adjust based on your model's context length
    )
    
    # Set up sampling parameters
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        # top_p=args.top_p
    )
    
    # Process each CSV file
    print(f"\nWill process {len(input_csvs)} CSV file(s): {', '.join(input_csvs)}")
    
    for i, input_csv in enumerate(input_csvs, 1):
        print(f"\n[{i}/{len(input_csvs)}] Starting processing of: {input_csv}")
        process_single_csv(llm, tokenizer, input_csv, args, sampling_params)
    
    print(f"\n{'='*60}")
    print(f"Successfully processed {len(input_csvs)} CSV file(s).")
    print(f"{'='*60}")

def run():
    main()

if __name__ == "__main__":
    main()