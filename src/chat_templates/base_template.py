""" """

from abc import ABC, abstractmethod
from typing import List, Union, Dict, Text, Any
from registrable import Registrable


class BaseTemplate(Registrable, ABC):
    """ """
    def __init__(
        self,
    ):
        """ """
        super().__init__()
        
    @abstractmethod
    def get_prompt_template(self, **kwargs) -> Union[List[Dict[Text, Any]], Text]:
        """ """
        
        raise NotImplementedError("This method must be implemented in the derived class.")
    
    @abstractmethod
    def get_completion_template(self, **kwargs) -> Union[List[Dict[Text, Any]], Text]:
        """ """
        
        raise NotImplementedError("This method must be implemented in the derived class.")