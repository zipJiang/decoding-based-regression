"""Encoder Prob Scorer
"""

import numpy as np
from typing import Dict, Text, Any
from registrable import Registrable, Lazy
from transformers.pipelines import Pipeline
from transformers import pipeline
from ..chat_templates import BaseTemplate
from ..utils.transforms import _inverse_sigmoid_unli
from overrides import overrides
from .base_prob_scorer import BaseProbScorer


@BaseProbScorer.register("encoder-based")
class EncoderProbScorer(BaseProbScorer):

    def __init__(
        self,
        model_name: Text
    ):
        """ """
        self._model_name = model_name
        self._pipeline = pipeline("text-classification", model="Zhengping/roberta-large-unli", device=0)
        
    @overrides
    def _score(self, pair: Dict[Text, Any]) -> np.ndarray:
        """ """
        # inputs = {
        #     "premise": pair["body"]["premise"],
        #     "hypothesis": pair["body"]["hypothesis"]
        # }
        result = self._pipeline({
            "text": pair["body"]["premise"],
            "text_pair": pair["body"]["hypothesis"]
        })

        return _inverse_sigmoid_unli(np.array(result["score"], dtype=np.float32)) / 10000