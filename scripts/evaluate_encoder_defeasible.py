"""
"""

import click
from transformers.pipelines import pipeline
from tqdm import tqdm
import evaluate
import datasets


def main():
    """ """

    pipe = pipeline("text-classification", model="Zhengping/roberta-large-unli", device=0)
    metric = evaluate.load("accuracy")
    
    for subset in ("snli", "atomic"):
        test_set = datasets.load_dataset("tasksource/defeasible-nli", subset, split="test")
        inputs = [
            {
                "text": datapiece["Premise"],
                "text_pair": datapiece["Hypothesis"]
            }
            for datapiece in test_set
        ]

        update_inputs = [
            {
                "text": datapiece["Premise"] + " " + datapiece["Update"],
                "text_pair": datapiece["Hypothesis"]
            }
            for datapiece in test_set
        ]

        results = pipe(tqdm(inputs))
        update_results = pipe(tqdm(update_inputs))

        preds = [int(u['score'] - r['score'] > 0) for r, u in zip(results, update_results)]
        refs = [int(t == "strengthener") for t in test_set['UpdateType']]
        
        # reference:
        # snli: 0.779
        # atomic: 0.725
        
        print(f"{subset} accuracy: {metric.compute(predictions=preds, references=refs)}")
        
if __name__ == "__main__":
    main()