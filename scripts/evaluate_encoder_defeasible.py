"""
"""

import click
import json
from scipy.stats import pearsonr, spearmanr
import numpy as np
from transformers.pipelines import pipeline
from transformers.pipelines.pt_utils import KeyPairDataset
from tqdm import tqdm
import evaluate
import datasets


def main():
    """ """

    pipe = pipeline("text-classification", model="Zhengping/roberta-large-unli", device=0)
    metric = evaluate.load("accuracy")
    
    # for subset in ("snli", "atomic"):
    #     test_set = datasets.load_dataset("tasksource/defeasible-nli", subset, split="test")
    #     inputs = [
    #         {
    #             "text": datapiece["Premise"],
    #             "text_pair": datapiece["Hypothesis"]
    #         }
    #         for datapiece in test_set
    #     ]

    #     update_inputs = [
    #         {
    #             "text": datapiece["Premise"] + " " + datapiece["Update"],
    #             "text_pair": datapiece["Hypothesis"]
    #         }
    #         for datapiece in test_set
    #     ]

    #     results = pipe(tqdm(inputs))
    #     update_results = pipe(tqdm(update_inputs))

    #     preds = [int(u['score'] - r['score'] > 0) for r, u in zip(results, update_results)]
    #     refs = [int(t == "strengthener") for t in test_set['UpdateType']]
        
    #     # reference:
    #     # snli: 0.779
    #     # atomic: 0.725
        
    #     print(f"{subset} accuracy: {metric.compute(predictions=preds, references=refs)}")
    
    # HellaSwag
    # dataset = datasets.load_dataset("Rowan/hellaswag", split="validation")
    # dataset = dataset.filter(lambda x: x['label'] in ['0', '1', '2', '3'])
    # labels = np.array([int(l) for l in dataset['label']], np.int64)
    
    # dataset = dataset.map(lambda examples: {
    #     "text": [
    #         ctx_a
    #         for ctx_a, ctx_b in zip(examples['ctx_a'], examples['ctx_b'])
    #         for eds in examples['endings']
    #         for ed in eds
    #     ],
    #     "text_pair": [
    #         ctx_b + " " + ed
    #         for ctx_a, ctx_b in zip(examples['ctx_a'], examples['ctx_b'])
    #         for eds in examples['endings']
    #         for ed in eds
    #     ],
    # }, remove_columns=dataset.column_names, batched=True)
    
    # inputs = [
    #     {
    #         "text": str(item['ctx_a']),
    #         "text_pair": str(item['ctx_b']) + " " + str(ending),
    #     } for item in dataset for ending in item['endings']
    # ]
    # results = [pipe(item) for item in tqdm(inputs)]
    
    # # reshape
    # preds = np.argmax(np.array([r['score'] for r in results], dtype=np.float32).reshape(-1, 4), axis=1)
    # print(f"hellaswag accuracy: {metric.compute(predictions=preds, references=labels)}")
    
    
    # COPA
    # dataset = datasets.load_dataset("pkavumba/balanced-copa", split="test")
    # inputs = [
    #     {"text": item['premise'], "text_pair": choice} if item['question'] == "effect" else
    #     {"text_pair": item['premise'], "text": choice}
    #     for item in dataset for choice in (item['choice1'], item['choice2'])
    # ]
    # labels = dataset['label']
    # results = [pipe(item) for item in tqdm(inputs)]
    # preds = np.argmax(np.array([r['score'] for r in results], dtype=np.float32).reshape(-1, 2), axis=1)
    
    # print(f"copa accuracy: {metric.compute(predictions=preds, references=labels)}")
    
    # circa
    
    with open("data/circa.jsonl", 'r', encoding='utf-8') as file_:
        items = [json.loads(line) for line in file_]

    inputs = [
        {
            "text": (item['context'][:-1] + "," if item['context'].endswith('.') else item['context']) + " and X asks the question: " + item['question'],
            "text_pair": "Y means \'Yes\' with the answer: " + item['answer']
        } for item in items
    ]
    labels = [item['plausibility'] for item in items]
    print("CIRCA SPEARMANR: ", spearmanr(labels, [item['score'] for item in tqdm(pipe(inputs))])[0])
        
        
if __name__ == "__main__":
    main()