""" """

import abc
import numpy as np
from registrable import Registrable
from typing import List, Text, Dict, Any


class BaseProbScorer(abc.ABC, Registrable):
    def __init__(self):
        super().__init__()

    def __call__(
        self,
        pairs: List[Dict[Text, Any]]
    ) -> np.ndarray:
        
        if isinstance(pairs, list):
            return self._batch_score(pairs)
        
        return self._score(pair)

    @abc.abstractmethod
    def _score(self, pair: Dict[Text, Any]) -> np.ndarray:
        """ """
        raise NotImplementedError()
    
    def _batch_score(self, pairs: List[Dict[Text, Any]]) -> np.ndarray:
        return np.array([self._score(pair) for pair in pairs])