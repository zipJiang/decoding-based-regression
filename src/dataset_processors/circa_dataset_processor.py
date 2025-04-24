"""Circa dataset processor.
"""

from overrides import overrides
import numpy as np
from ..chat_templates import BaseTemplate
from typing import Text, Union, Dict, Any
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
from .base_dataset_processor import BaseDatasetProcessor
from ..utils.transforms import _inverse_sigmoid_unli


@BaseDatasetProcessor.register("circa")
class CircaDatasetProcessor(BaseDatasetProcessor):
    """Circa dataset processor.
    """
    
    def __init__(
        self,
        template: BaseTemplate,
        datapath: Text,
        with_system_messages: bool = False,
    ):
        """Constructor for the CircaDatasetProcessor class.
        """
        super().__init__(template=template)
        self._with_system_messages = with_system_messages
        self._datapath = datapath
        
    @overrides
    def _process(self) -> Union[Dataset, DatasetDict]:
        """ """
        
        def _process_func(example) -> Dict[Text, Any]:
            """ """
            return {
                "prompt": self._template.get_prompt_template(
                    premise=(
                        example['context'][:-1] + "," if example['context'].endswith('.') else example['context']
                    ) + " and X asks the question: " + example['question'],
                    hypothesis= "Y means \'Yes\' with the answer: " + example['answer']
                ),
                "scores": example['plausibility']
            }
        
        dataset = load_dataset("json", data_files=self._datapath)["train"]
        dataset = dataset.map(
            _process_func,
            remove_columns=dataset.column_names if isinstance(dataset, Dataset) else dataset[list(dataset.keys())[0]].column_names
        )
        
        return DatasetDict({
            "test": dataset
        })