"""Prepare the UNLI dataset
"""

import numpy as np
from typing import Text, Union
from datasets import Dataset, DatasetDict, load_dataset
from overrides import overrides
from .base_dataset_processor import BaseDatasetProcessor
from ..utils.transforms import _inverse_sigmoid_unli


@BaseDatasetProcessor.register("unli")
class UNLIDatasetProcessor(BaseDatasetProcessor):
    def __init__(
        self,
        number_of_levels: int,
        with_system_messages: bool = False
    ):
        super().__init__()
        self._number_of_levels = number_of_levels
        self._with_system_messages = with_system_messages

    @overrides
    def _process(self) -> Dataset:
        return self._process_completion()
    
    def _label_bining(self, x: Union[float, np.ndarray]) -> int:
        
        # x is in [1, 10000]

        return max(
            min(
                int(x * self._number_of_levels / 10000), self._number_of_levels - 1
            ), 0
        )
    
    def _process_completion(self):
        dataset = load_dataset("Zhengping/UNLI")
        
        get_prompt = lambda x: ([{
            "role": "user",
            "content": "### Question: Given the premise \"{premise}\", how likely is it that the hypothesis \"{hypothesis}\" is true?\n\n".format(
                premise=x["premise"],
                hypothesis=x["hypothesis"]
            )}
        ])
        
        get_completion = lambda x: ([{
            "role": "assistant",
            "content": f"### Answer: <|label_level_{self._label_bining(_inverse_sigmoid_unli(x['label']))}|>"
        }])
        
        dataset = dataset.map(lambda example: {
            "prompt": get_prompt(example),
            "completion": get_completion(example),
            "scores": example["label"],
        }, remove_columns=[
            "premise",
            "hypothesis",
            "label",
            "snli-label"
        ])

        return dataset