"""Calling the to-mermaid function to generate
graph visualizations
"""

import json
from src.schema_workers.schema import Schema


def main():
    """ """
    with open(
        "/scratch/bvandur1/zjiang31/decoding-based-regression/task_outputs/struct_eval_results/sft-regression/Qwen2.5-14B-Instruct::nl=10::temp=1.0::reverse_kl=0::std=0.1::lsf=0.1::trust/bird-com2sense.jsonl",
        'r', encoding='utf-8'
    ) as file_:
        data = json.load(file_)
        # items = sorted([Schema.from_dict(json.loads(line)['schema']) for line in file_], key=lambda x: len(x.nodes), reverse=True)
        items = [Schema.from_dict(item['schema']) for item in data['outcomes']]
        print(items[10].to_mermaid())
        
        
if __name__ == "__main__":
    main()