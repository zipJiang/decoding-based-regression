"""
"""

import torch
from trl import DataCollatorForCompletionOnlyLM
from typing import Union, Any


class DataCollatorForSFTRegressionMixin:
    """Giving whatever the model outputs, this data collator will
    append a field for the scores.
    """
    
    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        """ """
        # We have to apply this hack to remove text fileds until
        # trl 0.15.0 has this fixed (remove_columns).
        
        for e in examples:
            e.pop('text', None)
            e.pop('messages', None)
        
        batch = super().torch_call(examples)
        # TODO: Making this more general and compatible with different data-types
        batch['scores'] = torch.tensor([example['scores'] for example in examples], dtype=torch.bfloat16)
        
        return batch
    
    
class DataCollatorForCompletionRegression(DataCollatorForSFTRegressionMixin, DataCollatorForCompletionOnlyLM):
    ...