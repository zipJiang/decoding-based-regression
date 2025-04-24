"""Create a datacollator where the token label is a soft distribution over the vocabs.
"""

import warnings
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import (
    Union, Optional, Any
)
from ..rank_dicts import BaseRankDict, SingleLabelRankDict
from .data_collator_for_regression import DataCollatorForSFTRegressionMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from transformers.data.data_collator import (
    DataCollatorMixin,
    pad_without_fast_tokenizer_warning
)
from trl import DataCollatorForCompletionOnlyLM
from ..utils.transforms import _discretize_gaussian

class DataCollatorForSingleTokenSoftLM(
    DataCollatorForSFTRegressionMixin,
    DataCollatorForCompletionOnlyLM
):
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.0 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """
    
    def __init__(
        self,
        *args,
        sigma: Optional[float] = 0.005,
        label_smoothing_factor: Optional[float] = 0.0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.sigma = sigma
        self.label_smoothing_factor = label_smoothing_factor
        self.rank_dict = SingleLabelRankDict.from_tokenizer(self.tokenizer)
        _rank_dict = self.rank_dict.get_rank_dict(tokenizer=self.tokenizer)
        tuples = sorted(_rank_dict.items(), key=lambda x: x[1], reverse=False)
        
        self.level_ids = torch.tensor([t[0] for t in tuples], dtype=torch.int64)
        self.levels = np.array([t[1] for t in tuples])
    
    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        batch = super().torch_call(examples)
        
        # at this point we have the input_ids, attention_mask, and labels
        # input_ids: [batch_size, seq_len]
        # attention_mask: [batch_size, seq_len]
        # labels: [batch_size, seq_len]
        # scores: [batch_size]
        assert 'scores' in examples[0], "The scores field is not found in the examples."
        # from probs create a soft distribution [batch_size, seq_len, vocab_size]
        labels = batch.pop("labels")
        mask = ~labels.eq(self.ignore_index)
        first_non_ignore_indices = torch.argmax(mask.int(), dim=1)  # [batch_size]
        # for those with ignored_index, convert to 0 in labels
        # We need to compute soft-labels from score
        vocab_size = len(self.tokenizer.get_vocab())
            
        _labels = labels.masked_fill(~mask, 0)
        soft_labels = torch.nn.functional.one_hot(_labels, num_classes=vocab_size).to(dtype=torch.float16)
            
        
        if not isinstance(examples[0]['scores'], list):
            scores = np.array([example['scores'] for example in examples])
            probs = _discretize_gaussian(
                mean=scores / 10000,
                std=self.sigma,
                levels=self.levels[np.newaxis, :]
            )  # [batch_size, num_levels]
            
        else:
            # We only need to load and insert the labels
            probs = np.array([example['scores'] for example in examples])
        
        soft_labels[
            torch.arange(soft_labels.shape[0]).unsqueeze(-1),
            first_non_ignore_indices.unsqueeze(-1),
            self.level_ids,
        ] = torch.tensor(probs, dtype=torch.float16)

        # renormalize with the label_smoothing_factor
        soft_labels = soft_labels * (1 - self.label_smoothing_factor) + self.label_smoothing_factor / vocab_size

        # again mask out the ignored_index
        batch['labels'] = soft_labels.masked_fill(~mask.unsqueeze(-1), self.ignore_index)
        
        return batch