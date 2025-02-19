"""
"""

try:
    import ujson as json
except ImportError:
    import json
from typing import Union, Text, List, Dict, Any
from datasets import Dataset, DatasetDict
from overrides import overrides
from ..chat_templates import BaseTemplate
from .base_dataset_processor import BaseDatasetProcessor
from .unli_dataset_processor import UNLIDatasetProcessor


@BaseDatasetProcessor.register("multi-premise")
class MultiPremiseProcessor(UNLIDatasetProcessor):
    """ """
    def __init__(
        self,
        number_of_levels: int,
        datapath: Union[Text, List[Text]],
        template: BaseTemplate,
        with_system_messages: bool = False
    ):
        """ """
        super().__init__(
            number_of_levels=number_of_levels,
            template=template,
            with_system_messages=with_system_messages
        )
        self._datapath = datapath if isinstance(datapath, list) else [datapath]
        
    @overrides
    def _process(self) -> Union[Dataset, DatasetDict]:
        """ """
        
        def _process_datapiece(datapiece) -> Dict[Text, Any]:
            """ """
            
            return {
                "premise": ' '.join(datapiece["premise"]),
                "hypothesis": datapiece["hypothesis"],
                "label": datapiece["label"]
            }
        
        all_items = []
        
        for dp in self._datapath:
            with open(dp, 'r', encoding='utf--8') as file_:
                for line in file_:
                    datapiece = _process_datapiece(json.loads(line))
                    all_items.append(datapiece)

        # dataset =  DatasetDict({
        #     "test": Dataset.from_list(all_items)
        # })
        
        return DatasetDict({
            "test": super()._process_completion(dataset=Dataset.from_list(all_items))
        })