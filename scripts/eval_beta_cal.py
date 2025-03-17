"""This experiment shows that isotonic regression is useful
in mapping the probability distribution.
"""

import click
import numpy as np
# from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from betacal import BetaCalibration
try:
    import ujson as json
except ImportError:
    import json
import os


@click.command()
@click.option("--data-path", type=str, help="Path to the data directory.")
def main(data_path):
    """ """
    
    for subdir in os.listdir(data_path):
        if not os.path.isdir(os.path.join(data_path, subdir)):
            continue
        
        with open(os.path.join(data_path, subdir, "unli-00000000.jsonl"), 'r', encoding='utf-8') as file_:
            items = [json.loads(line) for line in file_]
            
        # split by split
        test = list(filter(
            lambda x: x['processed::probability'] is not None,
            [item for item in items if item['split'] == 'test']
        ))
        dev = list(filter(
            lambda x: x['processed::probability'] is not None,
            [item for item in items if item['split'] == 'validation']
        ))

        # split by label
        # bc = BetaCalibration(parameters="abm")
        # bc.fit(
        #     np.array([d['processed::probability'] for d in dev], dtype=np.float32),
        #     np.array([d['label'] for d in dev], dtype=np.float32)
        # )
        # lr = LogisticRegression(C=9999999999)
        ist = IsotonicRegression()
        ist.fit(
            np.array([d['processed::probability'] for d in dev], dtype=np.float32).reshape(-1, 1),
            np.array([d['label'] for d in dev], dtype=np.float32)
        )
        
        pr = ist.predict(opr := np.array([t['processed::probability'] for t in test], dtype=np.float32))
        scores = np.array([t['label'] for t in test], dtype=np.float32)

        print("=== Results ===")
        print("Datsaset: ", subdir)
        print(f"MSE-before: {((opr - scores) ** 2).mean():.4f}")
        print(f"MSE: {((pr - scores) ** 2).mean():.4f}")
        print()
            

if __name__ == "__main__":
    main()