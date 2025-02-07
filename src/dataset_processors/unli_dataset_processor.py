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
        is_chat: bool,
        number_of_levels: int,
        with_system_messages: bool = False
    ):
        super().__init__()
        self._is_chat = is_chat
        self._number_of_levels = number_of_levels
        self._with_system_messages = with_system_messages

    @overrides
    def _process(self) -> Dataset:
        
        if self._is_chat:
            return self._process_chat()
        
        return self._process_completion()
    
    def _label_bining(self, x: Union[float, np.ndarray]) -> int:
        
        # x is in [1, 10000]

        return max(
            min(
                int(x * self._number_of_levels / 10000), self._number_of_levels - 1
            ), 0
        )
    
    def _process_chat(self):
        dataset = load_dataset("Zhengping/UNLI")

        get_prompt = lambda x: (
            "Given the premise \"{premise}\", how_likely is it that the hypothesis \"{hypothesis}\" is true?".format(
                premise=x["premise"],
                hypothesis=x["hypothesis"]
            )
        )
        
        get_completion = lambda x: (
            f"<|label_level_{self._label_bining(_inverse_sigmoid_unli(x['label']))}|>"
        )
        
        dataset = dataset.map(lambda example: {
            "messages": ([
                {
                    "role": "system",
                    "content": "You are a careful and helpful agent that try to communicate your probabilistic beliefs on the likelihood of a hypothesis being true."
                }
            ] if self._with_system_messages else []) + [
                {
                    "role": "user",
                    "content": get_prompt(example)
                },
                {
                    "role": "assistant",
                    "content": get_completion(example)
                }
            ]
        }, remove_columns=[
            "premise",
            "hypothesis",
            "label",
            "snli-label"
        ])
        
        return dataset
        
    def _process_completion(self):
        dataset = load_dataset("Zhengping/UNLI")
        
        get_prompt = lambda x: (
            "Given the premise \"{premise}\", how_likely is it that the hypothesis \"{hypothesis}\" is true?\n\n".format(
                premise=x["premise"],
                hypothesis=x["hypothesis"]
            )
        )
        
        get_completion = lambda x: (
            f"<|label_level_{self._label_bining(_inverse_sigmoid_unli(x['label']))}|>"
        )
        
        dataset = dataset.map(lambda example: {
            "prompt": get_prompt(example),
            "completion": get_completion(example)
        }, remove_columns=[
            "premise",
            "hypothesis",
            "label",
            "snli-label"
        ])

        return dataset