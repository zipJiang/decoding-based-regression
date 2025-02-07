""" """

from abc import ABC
from registrable import Registrable
from typing import Union
from datasets import Dataset, DatasetDict


class BaseDatasetProcessor(ABC, Registrable):
    def __init__(self):
        super().__init__()
        
    def __call__(self) -> Union[Dataset, DatasetDict]:
        
        return self._process()
        
    def _process(self) -> Union[Dataset, DatasetDict]:
        """ """
        raise NotImplementedError("This method must be implemented in the derived class.")