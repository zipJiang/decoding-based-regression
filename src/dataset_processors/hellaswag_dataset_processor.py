
from overrides import overrides
import numpy as np
from ..chat_templates import BaseTemplate
from typing import Text, Union, Dict, Any
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
from .base_dataset_processor import BaseDatasetProcessor
from ..utils.transforms import _inverse_sigmoid_unli


@BaseDatasetProcessor.register("hellaswag")
class HellaSWAGDatasetProcessor(BaseDatasetProcessor):
    """ """
    def __init__(
        self,
        template: BaseTemplate,
        with_system_messages: bool = False,
    ):
        """ """
        super().__init__(template=template)
        self._with_system_messages = with_system_messages
        
    @overrides
    def _process(self) -> Union[Dataset, DatasetDict]:
        """ """

        def _process(example):

            return {
                "prompt_a": self._template.get_prompt_template(
                    premise=example['ctx_a'],
                    hypothesis=example["ctx_b"] + " " + example['endings'][0]
                ),
                "prompt_b": self._template.get_prompt_template(
                    premise=example['ctx_a'],
                    hypothesis=example["ctx_b"] + " " + example['endings'][1]
                ),
                "prompt_c": self._template.get_prompt_template(
                    premise=example['ctx_a'],
                    hypothesis=example["ctx_b"] + " " + example['endings'][2]
                ),
                "prompt_d": self._template.get_prompt_template(
                    premise=example['ctx_a'],
                    hypothesis=example["ctx_b"] + " " + example['endings'][3]
                ),
                "label": int(example['label'])
            }
        
        dataset = load_dataset("Rowan/hellaswag", split="validation")
        dataset = dataset.filter(lambda x: x['label'] in ["0", "1", "2", "3"])
        dataset = dataset.map(
            _process,
            remove_columns=dataset.column_names if isinstance(dataset, Dataset) else dataset[list(dataset.keys())[0]].column_names
        )
        
        return DatasetDict({
            "test": dataset
        })