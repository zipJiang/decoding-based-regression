"""Calling the to-mermaid function to generate
graph visualizations
"""

import json
from src.schema_workers.schema import Schema


def main():
    """ """
    with open("data/graph/bird/com2sense-decomposed.jsonl", 'r', encoding='utf-8') as file_:
        items = sorted([Schema.from_dict(json.loads(line)['schema']) for line in file_], key=lambda x: len(x.nodes), reverse=True)
        print(items[1000].to_mermaid())
        
        
if __name__ == "__main__":
    main()