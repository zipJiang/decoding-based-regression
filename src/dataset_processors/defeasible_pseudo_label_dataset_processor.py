"""This processor takes both the UNLI + Defeasible NLI
as well as the pseudo-labeled subset.
"""

import random
import os
from tqdm import tqdm
from sklearn.isotonic import IsotonicRegression
from ..utils.transforms import _inverse_sigmoid_unli
try:
    import ujson as json
except ImportError:
    import json
from overrides import overrides
import numpy as np
from ..chat_templates import BaseTemplate
from typing import Text, Union, List, Text, Any, Dict, Literal
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
from .base_dataset_processor import BaseDatasetProcessor
from ..utils.transforms import _inverse_sigmoid_unli


@BaseDatasetProcessor.register("defeasible-pseudo-label")
class DefeasiblePseudoLabelDatasetProcessor(BaseDatasetProcessor):
    
    _prob_field = "probability w/o judge"
    
    def __init__(
        self,
        template: BaseTemplate,
        delta_nli_subsets: Text | List[Text],
        data_dir: Text,
        dataset_names: List[Text],
        with_system_messages: bool = False,
        # down_sample: int = 1,
        up_sample: int = 1,
        down_sample: int = 1,
        seed: int = 42,
        mixed_dev: bool = False
    ):
        """Constructor for the DefeasibleNLIDatasetProcessor class.
        
        how the pseudo-labels should be weighted.
        """
        
        super().__init__(template=template)
        self._with_system_messages = with_system_messages
        self._dataset_names = dataset_names
        self._data_dir = data_dir
        self._mixed_dev = mixed_dev
        # self._upsample = upsample
        # self._down_sample = down_sample
        self._up_sample = up_sample
        self._down_sample = down_sample
        self._seed: int = seed
        self._delta_nli_subsets = delta_nli_subsets if isinstance(delta_nli_subsets, list) else [delta_nli_subsets]

    def _process_unli(self) -> DatasetDict:
        """We load the train and validation split of UNLI and group them
        """

        dataset: DatasetDict = load_dataset("Zhengping/UNLI")
        
        def _convert_unli(dataset: Dataset) -> Dataset:
            """ """
            dataset = dataset.shuffle(seed=self._seed)
            dataset = dataset.map(lambda x: {
                "pscores": _inverse_sigmoid_unli(x["label"]),
            }, remove_columns="label")
            length = len(dataset)
            if length % 2 == 1:
                length -= 1
            first_half = dataset.select(range(length // 2))
            second_half = dataset.select(range(length // 2, length, 1))

            assert len(first_half) == len(second_half), "The two splits should have the same length."

            # each split has
            # - premise
            # - hypothesis
            # - snli-label
            # - label
            
            first_half = first_half.map(lambda x: {
                "prompt": self._template.get_prompt_template(
                    premise=x["premise"],
                    hypothesis=x["hypothesis"]
                ),
                "completion": self._template.get_completion_template(
                    answer=f" <|label_level_0|>"
                ),
                "is_defeasible": 0,
                "scores": [x["pscores"] / 10000],
                "confidence": [1.0],
            }, remove_columns=["premise", "hypothesis", "pscores", "snli-label"])
            
            second_half = second_half.map(lambda x: {
                "alternate_prompt": self._template.get_prompt_template(
                    premise=x["premise"],
                    hypothesis=x["hypothesis"]
                ),
                "alternate_completion": self._template.get_completion_template(
                    answer=f" <|label_level_0|>"
                ),
                "alternate_scores": [x["pscores"] / 10000],
                "alternate_confidence": [1.0],
            }, remove_columns=["premise", "hypothesis", "pscores", "snli-label"])

            return concatenate_datasets([first_half, second_half], axis=1)
        
        return DatasetDict({
            # split: _convert_unli(dataset[split])
            split: _convert_unli(dataset[split])
            for split in ["train", "validation"]
        })
        
    def _process_defeasible_nli(self) -> DatasetDict:
        """Naturally convert defeasible NLI instances.
        """
        
        datasets = []
        
        for subset in self._delta_nli_subsets:
            dataset = load_dataset("tasksource/defeasible-nli", subset)
            dataset = dataset.map(lambda x: {
                "prompt": self._template.get_prompt_template(
                    premise=x['Premise'],
                    hypothesis=x['Hypothesis']
                ),
                "completion": self._template.get_completion_template(
                    answer=f" <|label_level_0|>"
                ),
                "alternate_prompt": self._template.get_prompt_template(
                    premise=x['Premise'] + " " + x['Update'],
                    hypothesis=x['Hypothesis']
                ),
                "alternate_completion": self._template.get_completion_template(
                    answer=f" <|label_level_0|>"
                ),
                "is_defeasible": 1,
                "scores": [0.0 if x["UpdateType"] == "strengthener" else 1.0],
                "alternate_scores": [1.0 if x["UpdateType"] == "strengthener" else 0.0],
                "confidence": [1.0],
                "alternate_confidence": [1.0],
            }, remove_columns=dataset['train'].column_names)

            datasets.append(dataset)
            
        return DatasetDict({
            "train": concatenate_datasets([d["train"] for d in datasets]),
            "validation": concatenate_datasets([d["validation"] for d in datasets])
        })
        
    def _process_hellaswag(self) -> DatasetDict:
        """hellaswag needs sampling.
        """
        random_obj = random.Random(self._seed)
        def _process(example) -> Dict[Text, Any]:
            
            label = int(example['label'])
            neg_candidates = [ending for eidx, ending in enumerate(example['endings']) if eidx != label]
            neg_candidate = random_obj.choice(neg_candidates)

            # randomize the directions
            direction = random_obj.randint(0, 1)
            
            if direction == 0:
                return {
                    "prompt": self._template.get_prompt_template(
                        premise=example['ctx_a'],
                        hypothesis=example["ctx_b"] + " " + example['endings'][label]
                    ),
                    "completion": self._template.get_completion_template(
                        answer=f" <|label_level_0|>"
                    ),
                    "alternate_prompt": self._template.get_prompt_template(
                        premise=example['ctx_a'],
                        hypothesis=example["ctx_b"] + " " + neg_candidate,
                    ),
                    "alternate_completion": self._template.get_completion_template(
                        answer=f" <|label_level_0|>"
                    ),
                    "is_defeasible": 1,
                    "scores": [1.0],
                    "alternate_scores": [0.0],
                    "confidence": [1.0],
                    "alternate_confidence": [1.0],
                }
            
            return {
                "prompt": self._template.get_prompt_template(
                    premise=example['ctx_a'],
                    hypothesis=example["ctx_b"] + " " + neg_candidate
                ),
                "completion": self._template.get_completion_template(
                    answer=f" <|label_level_0|>"
                ),
                "alternate_prompt": self._template.get_prompt_template(
                    premise=example['ctx_a'],
                    hypothesis=example["ctx_b"] + " " + example['endings'][label],
                ),
                "alternate_completion": self._template.get_completion_template(
                    answer=f" <|label_level_0|>"
                ),
                "is_defeasible": 1,
                "scores": [0.0],
                "alternate_scores": [1.0],
                "confidence": [1.0],
                "alternate_confidence": [1.0],
            }
        
        dataset = load_dataset("Rowan/hellaswag")
        dataset = DatasetDict({
            "train": dataset["train"],
            "validation": dataset["validation"]
        })
        dataset = dataset.filter(lambda x: x['label'] in ["0", "1", "2", "3"])
        return dataset.map(
            _process,
            remove_columns=dataset.column_names if isinstance(dataset, Dataset) else dataset[list(dataset.keys())[0]].column_names
        )
        
    def _process_synthetic(
        self
    ) -> DatasetDict:
        """ """
        
        def _fix_proba(ist_list: List[IsotonicRegression], items) -> List[List[float]]:
            probs = np.array([[s if s is not None else -1 for s in item[self._prob_field]] for item in items], dtype=np.float32)
            masks = probs > -1e-6
            probs[~masks] = 0.5
            # split probs of shape (a, b) into bs (a, 1)
            prob_splitted = np.split(probs, probs.shape[1], axis=1)
            transformed = np.stack([
                np.clip(
                    ist.predict(pbs),
                    a_min=0,
                    a_max=1
                )
                for ist, pbs in zip(ist_list, prob_splitted)
            ], axis=1)

            assert transformed.shape == probs.shape, "The shape of the transformed array should be the same as the original."
            transformed[~masks] = -1
            
            return transformed.tolist()
        
        def _tune_ist(index: int) -> IsotonicRegression:
            ist = IsotonicRegression()
            items = [item for item in filter(lambda x: x[self._prob_field][index] is not None, data)]
            probs = np.array([item[self._prob_field][index] for item in items]).reshape(-1, 1)
            labels = _inverse_sigmoid_unli(np.array([item['label'] for item in items])) / 10000
            ist.fit(probs, labels)
            
            return ist
        
        with open(os.path.join(self._data_dir, "unli-00000000_0.5.json"), 'r', encoding='utf-8') as file_:
            data = json.load(file_)
        ist_list = [_tune_ist(i) for i in range(len(data[0][self._prob_field]))]

        all_items = []

        for dataset_name in self._dataset_names:
            shard_index = 0
            while data_path := os.path.join(self._data_dir, f"{dataset_name}-{shard_index:08d}_0.5.json"):
                if not os.path.exists(data_path):
                    break
                
                with open(data_path, 'r', encoding='utf-8') as file_:
                    all_items.extend(json.load(file_))
                
                shard_index += 1
                
        fixed_probs = _fix_proba(ist_list, all_items)
        assert len(fixed_probs) == len(all_items), "The number of items should be the same."
        
        def _valid(item, fp):
            return ('confidence' not in item) or (
                (item['confidence'] is not None) and (len(item['confidence']) == len(fp))
            )
        
        all_items = [
            { 
                # "premise": item['premise'],
                # "hypothesis": item['hypothesis'],
                "split": item['split'],
                "prompt": self._template.get_prompt_template(
                    premise=item['premise'],
                    hypothesis=item['hypothesis']
                ),
                "completion": self._template.get_completion_template(
                    answer=f" <|label_level_0|>"
                ),
                "scores": fixedp,
                "confidence": item.get("confidence", [1.0] * len(fixedp)),
            } for item, fixedp in zip(all_items, fixed_probs)
            if _valid(item, fixedp)
        ]

        print("Number of items:", len(all_items))

        train_items = [{k: v for k, v in item.items() if k != 'split'} for item in all_items if item['split'].startswith('train')]
        validation_items = [{k: v for k, v in item.items() if k != 'split'} for item in all_items if item['split'].startswith('test')]
        
        random_obj = random.Random(self._seed)
        random_obj.shuffle(train_items)
        random_obj.shuffle(validation_items)
        
        train_items = [
            {
                **t,
                "alternate_prompt": tt['prompt'],
                "alternate_completion": tt['completion'],
                "alternate_scores": tt['scores'],
                "alternate_confidence": tt['confidence'],
                "is_defeasible": 0,
            } for t, tt in zip(train_items[:len(train_items) // 2], train_items[len(train_items) // 2:])
        ]
        
        validation_items = [
            {
                **t,
                "alternate_prompt": tt['prompt'],
                "alternate_completion": tt['completion'],
                "alternate_scores": tt['scores'],
                "alternate_confidence": tt['confidence'],
                "is_defeasible": 0,
            } for t, tt in zip(validation_items[:len(validation_items) // 2], validation_items[len(validation_items) // 2:])
        ]
        
        return DatasetDict({
            "train": Dataset.from_list(train_items),
            "validation": Dataset.from_list(validation_items)
        })
    
    @overrides
    def _process(self) -> Union[Dataset, DatasetDict]:
        """
        This datasets will have the following columns:
        - prompt:
        - completion:
        - alternate_prompt:
        - alternate_completion:
        - confidence: -- same size of scores
        - alternate_confidence: -- same size of alternate_scores
        - is_defeasible: int (0 or 1) -- 1 for defeasible, 0 for non-defeasible
        - scores: [float] -- while we don't have scores for defeasible-nli, we assign 0.5 and does not use it for training.
        - alternate_scores: [float] -- while we don't have scores for defeasible-nli, we assign 0.5 and does not use it for training.
        """
        
        unli_dataset = self._process_unli()
        defeasible_nli = self._process_defeasible_nli()
        hellaswag = self._process_hellaswag()
        pseudo_labeled = self._process_synthetic()

        print(pseudo_labeled)
        
        # each of them should have train and validation,
        # concatenate each split (shuffle done at training time)

        return DatasetDict({
            "train": concatenate_datasets(
                [unli_dataset["train"], defeasible_nli["train"], hellaswag["train"]] * self._up_sample + [pseudo_labeled["train"].select(range(len(pseudo_labeled["train"]) // self._down_sample))]
            ),
            "validation": (
                concatenate_datasets(
                    [
                        unli_dataset["validation"],
                        defeasible_nli["validation"],
                        hellaswag["validation"]
                    ] + [pseudo_labeled["validation"]]
                ) if self._mixed_dev
                else unli_dataset["validation"]
            )
        })