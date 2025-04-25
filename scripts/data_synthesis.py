import os
import argparse
import json
from src.data_synthesis.prompts import ProbsExtractorPrompt, ProbsComparePrompt
from src.data_synthesis.uncertainty_filter import UncertaintyFilter

def parse_args():
    
    parser = argparse.ArgumentParser(description="Uncertainty Filter")
    parser.add_argument(
        "--data_name", 
        type=str, 
        nargs='+',  # Accept multiple values as a list
        default=["unli-00000000.jsonl"], 
        help="List of data file names"
    )
    parser.add_argument("--discrepancy", type=float, default=0.5, help="Discrepancy threshold")
    parser.add_argument("--judge_dir", type=str, default="_judge", help="Directory for judge files")
    return parser.parse_args()

def main():
    args = parse_args()
    data_name = args.data_name
    for data in data_name:
        print(f"Processing {data}")
        filter = UncertaintyFilter(
            ds_path=f"pseudo-labeled/DeepSeek-R1-Distill-Qwen-32B/{data}",
            qwen_path=f"pseudo-labeled/Qwen2.5-72B-Instruct-AWQ/{data}",
            qwq_path=f"pseudo-labeled/QwQ-32B/{data}",
            llama_path=f"pseudo-labeled/Llama-3.3-70B-Instruct/{data}",
            discrepancy=args.discrepancy,
            judge_dir=args.judge_dir
        )
        filter.score_judge()
        filter.score_aggregation()

if __name__ == "__main__":
    main()