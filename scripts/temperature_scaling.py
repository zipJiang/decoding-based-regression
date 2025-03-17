"""Check the effect of temperature scaling on the model's calibration.
"""

import click
import json
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr


@click.command()
@click.option('--eval-dict-path', '-e', type=click.Path(exists=True), help='Path to the evaluation dictionary')
def main(
    eval_dict_path
):
    """ """
    
    with open(eval_dict_path):
        pass