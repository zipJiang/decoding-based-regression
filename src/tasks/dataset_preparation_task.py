""" """

import os
from datasets import DatasetDict, Dataset
from overrides import overrides
from typing import Text
from tasker import BaseTask
from ..dataset_processors import BaseDatasetProcessor


@BaseTask.register("dataset-preparation")
class DatasetPreparationTask(BaseTask):
    """ """
    
    __VERSION__ = "0.0.5"

    def __init__(
        self,
        dataset_processor: BaseDatasetProcessor,
        output_dir: Text,
    ):
        """ """
        super().__init__(output_dir=output_dir)
        self._dataset_processor = dataset_processor
        
    @overrides
    def _run(self):
        """ """
        dataset = self._dataset_processor()
        # dataset.save_to_disk(os.path.join(
        #     self._output_dir,
        #     "dataset"
        # ))
        if isinstance(dataset, DatasetDict):
            print(dataset['train'][:5])
        elif isinstance(dataset, Dataset):
            print(dataset[:5])
        
        # This is best effort based review
        
        return dataset
    
    def _write(self, outputs):
        
        outputs.save_to_disk(os.path.join(
            self._output_dir,
            "dataset"
        ))