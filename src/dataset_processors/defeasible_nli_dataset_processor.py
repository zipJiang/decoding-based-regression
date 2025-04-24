"""Defeasible NLI dataset processor.
"""

from overrides import overrides
import numpy as np
from ..chat_templates import BaseTemplate
from typing import Text, Union
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
from .base_dataset_processor import BaseDatasetProcessor
from ..utils.transforms import _inverse_sigmoid_unli


@BaseDatasetProcessor.register("defeasible-nli")
class DefeasibleNLIDatasetProcessor(BaseDatasetProcessor):
    """Defeasible NLI dataset processor.
    """
    
    def __init__(
        self,
        template: BaseTemplate,
        subset: Text,
        with_system_messages: bool = False,
    ):
        """Constructor for the DefeasibleNLIDatasetProcessor class.
        """
        super().__init__(template=template)
        self._with_system_messages = with_system_messages
        self._subset = subset
        
    @overrides
    def _process(self) -> Union[Dataset, DatasetDict]:
        """ """
        get_prompt = lambda x: self._template.get_prompt_template(
            premise=x['Premise'],
            hypothesis=x['Hypothesis']
        )

        get_update_prompt = lambda x: self._template.get_prompt_template(
            premise=x['Premise'] + " " + x['Update'],
            hypothesis=x['Hypothesis']
        )
        
        dataset = load_dataset("tasksource/defeasible-nli", self._subset)
        dataset = dataset.map(
            lambda x: {
                "is_strengthener": x["UpdateType"] == "strengthener",
                "prompt": get_prompt(x),
                "update_prompt": get_update_prompt(x),
                # TODO: add completion processing as well.
            }, remove_columns=dataset.column_names if isinstance(dataset, Dataset) else dataset[list(dataset.keys())[0]].column_names
        )

        return dataset