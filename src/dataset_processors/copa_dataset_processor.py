"""COPA dataset processor.
"""

from overrides import overrides
import numpy as np
from ..chat_templates import BaseTemplate
from typing import Text, Union, Dict, Any
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
from .base_dataset_processor import BaseDatasetProcessor
from ..utils.transforms import _inverse_sigmoid_unli


@BaseDatasetProcessor.register("copa")
class COPADatasetProcessor(BaseDatasetProcessor):
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
        
        train_data = load_dataset("pkavumba/balanced-copa", split="train")
        test_data = load_dataset("pkavumba/balanced-copa", split="test")
        dataset = concatenate_datasets([train_data, test_data])
        
        
        def _process_func(example) -> Dict[Text, Any]:
            """ """

            if example['question'] == 'effect':
                a_instance = self._template.get_prompt_template(
                    premise=example['premise'],
                    hypothesis=example['choice1']
                )
                b_instance = self._template.get_prompt_template(
                    premise=example['premise'],
                    hypothesis=example['choice2']
                )
                
            elif example['question'] == 'cause':
                a_instance = self._template.get_prompt_template(
                    premise=example['choice1'],
                    hypothesis=example['premise']
                )
                b_instance = self._template.get_prompt_template(
                    premise=example['choice2'],
                    hypothesis=example['premise']
                )
                
            else:
                raise ValueError("Invalid question type.")
            
            return {
                "prompt_a": a_instance,
                "prompt_b": b_instance,
                "more_likely_index": example['label']
            }
        
        dataset = dataset.map(
            _process_func,
            remove_columns=dataset.column_names if isinstance(dataset, Dataset) else dataset[list(dataset.keys())[0]].column_names
        )
        
        return DatasetDict({
            "test": dataset
        })