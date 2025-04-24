""" """

from typing import List, Union, Dict, Text, Any
from .base_template import BaseTemplate


@BaseTemplate.register("unli")
class UNLITemplate(BaseTemplate):
    """ """
    def __init__(
        self,
    ):
        """ """
        super().__init__()
        
    def get_prompt_template(self, **kwargs) -> Union[List[Dict[Text, Any]], Text]:
        """ """
        
        return [{
            "role": "user",
            "content": "### Question: Given the premise \"{premise}\", how likely is it that the hypothesis \"{hypothesis}\" is true?\n\n".format(
                **kwargs
            )}
        ]
    
    def get_completion_template(self, **kwargs) -> Union[List[Dict[Text, Any]], Text]:
        """ """
        
        is_completion = kwargs.pop("is_completion", False)
        
        return [{
            "role": "assistant",
            "content": "### Answer:{answer}".format(
                **kwargs
            ) if not is_completion else "### Answer:"
        }]