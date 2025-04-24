"""Inspired by learning to rank, we try to regularize with
learning to rank-loss on defeasible instances.

- Defeasible-NLI
- HellaSwag
"""

import random
from overrides import overrides
import numpy as np
from ..chat_templates import BaseTemplate
from typing import Text, Union, List, Text, Any, Dict
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
from .base_dataset_processor import BaseDatasetProcessor
from ..utils.transforms import _inverse_sigmoid_unli


@BaseDatasetProcessor.register("defeasible-training")
class DefeasibleTrainingDatasetProcessor(BaseDatasetProcessor):
    def __init__(
        self,
        template: BaseTemplate,
        delta_nli_subsets: Text | List[Text],
        with_system_messages: bool = False,
        seed: int = 42,
        mixed_dev: bool = False
    ):
        """Constructor for the DefeasibleNLIDatasetProcessor class.
        """
        super().__init__(template=template)
        self._with_system_messages = with_system_messages
        self._mixed_dev = mixed_dev
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
        
    
    @overrides
    def _process(self) -> Union[Dataset, DatasetDict]:
        """
        This datasets will have the following columns:
        - prompt:
        - completion:
        - alternate_prompt:
        - alternate_completion:
        - is_defeasible: int (0 or 1) -- 1 for defeasible, 0 for non-defeasible
        - scores: float -- while we don't have scores for defeasible-nli, we assign 0.5 and does not use it for training.
        - alternate_scores: float -- while we don't have scores for defeasible-nli, we assign 0.5 and does not use it for training.
        """
        
        unli_dataset = self._process_unli()
        defeasible_nli = self._process_defeasible_nli()
        hellaswag = self._process_hellaswag()
        
        # each of them should have train and validation,
        # concatenate each split (shuffle done at training time)

        return DatasetDict({
            "train": concatenate_datasets([unli_dataset["train"], defeasible_nli["train"], hellaswag["train"]]),
            "validation": (
                concatenate_datasets([unli_dataset["validation"], defeasible_nli["validation"], hellaswag["validation"]]) if self._mixed_dev
                else unli_dataset["validation"]
            )
        })