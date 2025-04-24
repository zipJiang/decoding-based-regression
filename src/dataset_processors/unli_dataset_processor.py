"""Prepare the UNLI dataset
"""

import numpy as np
from typing import Text, Union
from datasets import Dataset, DatasetDict, load_dataset
from overrides import overrides
from .base_dataset_processor import BaseDatasetProcessor
from ..chat_templates import UNLITemplate, BaseTemplate
from ..utils.transforms import _inverse_sigmoid_unli


@BaseDatasetProcessor.register("unli")
class UNLIDatasetProcessor(BaseDatasetProcessor):
    def __init__(
        self,
        number_of_levels: int,
        template: BaseTemplate,
        with_system_messages: bool = False,
    ):
        super().__init__(template=template)
        self._number_of_levels = number_of_levels
        self._with_system_messages = with_system_messages

    @overrides
    def _process(self) -> Union[Dataset, DatasetDict]:
        dataset = load_dataset("Zhengping/UNLI")
        return self._process_completion(dataset=dataset)
    
    def _label_bining(self, x: Union[float, np.ndarray]) -> int:
        
        # x is in [1, 10000]

        return max(
            min(
                int(x * self._number_of_levels / 10000), self._number_of_levels - 1
            ), 0
        )
    
    def _process_completion(self, dataset: Union[DatasetDict, Dataset]) -> Union[Dataset, DatasetDict]:
        
        # get_prompt = lambda x: ([{
        #     "role": "user",
        #     "content": "### Question: Given the premise \"{premise}\", how likely is it that the hypothesis \"{hypothesis}\" is true?\n\n".format(
        #         premise=x["premise"],
        #         hypothesis=x["hypothesis"]
        #     )}
        # ])
        
        # TODO: making this optional
        dataset = dataset.map(lambda x: {
            "pscores": _inverse_sigmoid_unli(x["label"]),
        }, remove_columns="label")
        
        get_prompt = lambda x: self._template.get_prompt_template(
            premise=x['premise'],
            hypothesis=x['hypothesis']
        )
        
        # get_completion = lambda x: ([{
        #     "role": "assistant",
        #     "content": f"### Answer: <|label_level_{self._label_bining(_inverse_sigmoid_unli(x['label']))}|>"
        # }])
        
        get_completion = lambda x: self._template.get_completion_template(
            answer=f" <|label_level_{self._label_bining(x['pscores'])}|>"
        )
        
        dataset = dataset.map(lambda example: {
            "prompt": get_prompt(example),
            "completion": get_completion(example),
            "scores": example["pscores"],
            # Remove column_names if is dataset otherwise DatsetDict has split names as column names
        }, remove_columns=dataset.column_names if isinstance(dataset, Dataset) else dataset[list(dataset.keys())[0]].column_names)

        return dataset