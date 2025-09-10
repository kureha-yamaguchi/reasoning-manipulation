"""
Script to filter model output generations based on StrongReject evaluator scores.

Usage:
    CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python -m utils.filter_datasets --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B
"""

import csv
import os
import argparse
from typing import List, Dict, Tuple, Any
from itertools import islice

import torch
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer
from strong_reject.evaluate import evaluate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter model generations depending on StrongReject evaluator scores"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        help="Model to use for generation"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/",
        help="Results parent directory"
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default="all_harmful_prompts_cot3_out5.csv",
        help="Dataset input CSV filename"
    )
    parser.add_argument(
        "--lower_threshold",
        type=float,
        default=0.15,
        help="Insert lower threshold for creating refusal dataset"
    )
    parser.add_argument(
        "--upper_threshold",
        type=float,
        default=0.85,
        help="Insert upper threshold for creating non-refusal dataset"
    )

    return parser.parse_args()


def filter_csv(
    model_name: str,
    evaluator_results: Dict[str, List[float]],
    cot_dataset: Dataset,
    lower_threshold: float,
    upper_threshold: float
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], int]:
    """
    Filter Chain-of-Thought responses based on StrongReject evaluator scores.
    
    This function processes evaluation scores in chunks (corresponding to multiple
    outputs from the same CoT), and classifies them as refusal or non-refusal
    based on whether all scores in a chunk meet the specified thresholds.
    
    Args:
        model_name: Name or path of the model (used to load tokenizer)
        evaluator_results: Dictionary containing evaluation scores under 'score' key
        cot_dataset: HuggingFace Dataset containing 'prompt' and 'cot' columns
        lower_threshold: Score threshold for refusal classification (all scores < threshold)
        upper_threshold: Score threshold for non-refusal classification (all scores > threshold)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Initialize containers for filtered data
    refusal = []
    non_refusal = []
    token_lengths = []

    scores = evaluator_results["score"]
    chunk_size = 5  # Number of output variations per CoT

    # Process scores in chunks corresponding to same CoT
    for start in range(0, len(scores), chunk_size):
        end = min(start + chunk_size, len(scores))
        same_cot_chunk = scores[start:end]

        # Determine classification based on all scores in chunk
        include_refusal = False
        include_nonrefusal = False
        
        if all(score < lower_threshold for score in same_cot_chunk):
            include_refusal = True
        elif all(score > upper_threshold for score in same_cot_chunk):
            include_nonrefusal = True
        else:
            print(f"Chunk starting at index {start} has not been filtered into refusal nor non-refusal datasets")
     
        # Add to appropriate dataset if criteria met
        if include_refusal:
            refusal.append({
                "prompt": cot_dataset["prompt"][start],
                "cot": cot_dataset["cot"][start]
            })
            # Calculate token lengths for all items in chunk for calculation of smallest_token_length 
            for i in range(start, end):
                tokens = tokenizer.encode(cot_dataset["cot"][i])
                token_lengths.append(len(tokens))
        elif include_nonrefusal:
            non_refusal.append({
                "prompt": cot_dataset["prompt"][start],
                "cot": cot_dataset["cot"][start]
            })
            # Calculate token lengths for all items in chunk for calculation of smallest_token_length
            for i in range(start, end):
                tokens = tokenizer.encode(cot_dataset["cot"][i])
                token_lengths.append(len(tokens))

    
    smallest_token_length = min(token_lengths)
    
    return refusal, non_refusal, smallest_token_length


def write_to_csv(filtered_data: List[Dict[str, str]], output_file: str) -> None:
    """
    Write filtered prompt-CoT pairs to a CSV file.
    
    Args:
        filtered_data: List of dictionaries containing 'prompt' and 'cot' keys
        output_file: Path to the output CSV file
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Write to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['prompt', 'cot']
        writer: csv.DictWriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in filtered_data:
            writer.writerow(row)
    
    print(f"Successfully exported {len(filtered_data)} rows to {output_file}")


def main() -> None:

    # Parse arguments and setup
    args: argparse.Namespace = parse_args()
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    model_name = args.model_name
    csv_path = os.path.join(args.results_dir, args.model_name, "dataset", args.input_csv)

    # Initialize data containers
    prompt = []
    cot = []  # CoT including </think> tag
    output = []  # Final outputs after </think>

    # Read CSV file and extract relevant columns
    print(f"Reading data from: {csv_path}")
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader: csv.DictReader = csv.DictReader(f)
        # for row in tqdm(islice(reader, 30), desc="Processing rows"):
        for row in tqdm(reader, desc="Processing rows"):
            prompt.append(row["prompt"])
            cot.append(row["cot"])
            output.append(row["output"])

    # Create output datasets for evaluation
    output_dataset: Dataset = Dataset.from_dict({
        "forbidden_prompt": prompt,
        "response": output
    })

    cot_dataset: Dataset = Dataset.from_dict({
        "prompt": prompt,
        "cot": cot
    })

    # Evaluate outputs using StrongReject
    print("Evaluating outputs with StrongReject...")
    evaluator_results = evaluate_dataset(
        output_dataset,
        ["strongreject_finetuned"]
    )

    # Filter based on evaluation scores
    print(f"Processed datasets: {len(prompt)} rows")
    refusal, non_refusal, smallest_token_length = filter_csv(
        model_name,
        evaluator_results,
        cot_dataset,
        args.lower_threshold,
        args.upper_threshold
    )
    
    print(f"Smallest token length: {smallest_token_length}")
    print(f"Refusal samples: {len(refusal)}")
    print(f"Non-refusal samples: {len(non_refusal)}")

    # Save filtered datasets
    output_refusal_path = os.path.join(
        args.results_dir,
        args.model_name,
        "dataset",
        f"refusal_{args.lower_threshold}.csv"
    )
    write_to_csv(refusal, output_file=output_refusal_path)

    output_nonrefusal_path = os.path.join(
        args.results_dir,
        args.model_name,
        "dataset",
        f"nonrefusal_{args.upper_threshold}.csv"
    )
    write_to_csv(non_refusal, output_file=output_nonrefusal_path)

    # Also save smallest_token_length
    token_length_path = os.path.join(
        args.results_dir,
        args.model_name,
        "dataset",
        "smallest_token_length.txt"
    )

    with open(token_length_path, 'w') as f:
        f.write(str(smallest_token_length))


if __name__ == "__main__":
    main()