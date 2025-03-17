"""Subsample a set of data points from a dataset.
"""

import click
from collections import defaultdict
import os
import random
try:
    import ujson as json
except ImportError:
    import json
    
    
@click.command()
@click.option(
    "--input-path",
    required=True,
    type=click.Path(exists=True),
)
@click.option(
    "--seed",
    required=False,
    type=int,
    default=2265
)
def main(
    input_path,
    seed
):
    """ """
    
    label_to_datalines = defaultdict(list)
    random_obj = random.Random(seed)

    with open(input_path, "r") as f:
        for ridx, row in enumerate(f):
            dl = json.loads(row)
            if "ground_truth" in dl:
                label_to_datalines[dl["ground_truth"]].append({**dl, "original_idx": ridx})
            else:
                # label is within the metadata field of schema.
                label_to_datalines[dl['schema']["metadata"]["answer"]].append({**dl, "original_idx": ridx})
                
    # calculate the number of minimum data points for each label
    min_count = min([len(datalines) for datalines in label_to_datalines.values()])
    num_samples = min(300, min_count)
    
    sampled_datalines = []
    
    for datalines in label_to_datalines.values():
        sampled_datalines.extend(random_obj.sample(datalines, num_samples))
    random_obj.shuffle(sampled_datalines)
    
    output_path = input_path.rsplit('.', 1)[0] + '-sampled.' + input_path.rsplit('.', 1)[1]
    with open(output_path, 'w') as f:
        for dl in sampled_datalines:
            f.write(json.dumps(dl) + '\n')
            
            
if __name__ == "__main__":
    main()