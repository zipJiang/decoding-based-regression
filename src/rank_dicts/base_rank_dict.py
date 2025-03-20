"""Class for inducing rank_dict from tokenizer
"""

import re
from abc import ABC, abstractmethod
from transformers import PreTrainedTokenizer
from typing import Dict, List, Text, Any
from registrable import Registrable


class BaseRankDict(Registrable, ABC):
    
    def __init__(
        self,
        rank_dict: Dict[Text, Any]
    ):
        self._rank_dict = rank_dict
        
    def __len__(self) -> int:
        return len(self._rank_dict)
        
    def get_rank_dict(self, tokenizer: PreTrainedTokenizer) -> Dict[int, Any]:
        return {tokenizer.convert_tokens_to_ids([token])[0]: value for token, value in self._rank_dict.items()}
    
    def to_tokenizer(self, tokenizer: PreTrainedTokenizer) -> PreTrainedTokenizer:
        """Augment tokenizer vocab with `rank_dict` IN-PLACE.
        """
        vocabs: List[Text] = self._rank_dict.keys()
        new_vocab = [vocab for vocab in vocabs if vocab not in tokenizer.get_vocab()]
        tokenizer.add_tokens(new_vocab)
        return tokenizer
        
    @classmethod
    @abstractmethod
    def from_tokenizer(cls, tokenizer: PreTrainedTokenizer) -> "BaseRankDict":
        """ """
        
        raise NotImplementedError("This method must be implemented in a child class.")
    
    
@BaseRankDict.register("single-label", constructor="from_tokenizer")
class SingleLabelRankDict(BaseRankDict):
    
    def __init__(
        self,
        rank_dict: Dict[Text, Any]
    ):
        super().__init__(rank_dict=rank_dict)
    
    @classmethod
    def from_tokenizer(cls, tokenizer: PreTrainedTokenizer) -> "SingleLabelRankDict":
        vocab = tokenizer.get_vocab()
        rank_dict = {}
        pattern = re.compile(r" <\|label_level_(\d+)\|>")
        
        for token in vocab.keys():
            match = pattern.match(token)
            if match:
                value = int(match.group(1))
                # normalized_value = value / (len(vocab) - 1)
                rank_dict[token] = value
                
        # normalize rank_values
        num_levels = max(rank_dict.values()) + 1
        for token in rank_dict.keys():
            rank_dict[token] = 1. / num_levels * (rank_dict[token] + 0.5)
        
        return cls(rank_dict=rank_dict)