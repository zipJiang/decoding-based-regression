"""Looking into responses that answer-label does not equal to the answer.
"""

import click
import json


@click.command()
@click.option("--input-path", type=click.Path(exists=True), help="Path to the input file.")
@click.option("--output-path", type=click.Path(), help="Path to the output file.")
def main(
    input_path,
    output_path
):
    """ """
    
    with open(input_path, 'r', encoding='utf-8') as file_:
        data = json.load(file_)['outcomes']
        
    incorrect_items = [item for item in data if item['answer_label'] != item['schema']['metadata']['answer']]
    print(f"Total number of incorrect items: {len(incorrect_items)}")
    
    
    with open(output_path, 'w', encoding='utf-8') as file_:
        json.dump(incorrect_items, file_, ensure_ascii=False, indent=4)
        
        
if __name__ == "__main__":
    main()