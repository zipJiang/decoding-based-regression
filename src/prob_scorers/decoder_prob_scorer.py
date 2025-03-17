""" """

import numpy as np
from typing import Dict, Text, Any
from registrable import Registrable, Lazy
from transformers.pipelines import Pipeline
from ..chat_templates import BaseTemplate
from overrides import overrides
from .base_prob_scorer import BaseProbScorer


@BaseProbScorer.register("decoder-based")
class DecoderProbScorer(BaseProbScorer):
    """ 
    """
    def __init__(
        self,
        template: BaseTemplate,
        pipeline: Lazy[Pipeline]
    ):
        """ """
        super().__init__()
        self._template = template
        self._pipeline = pipeline
        
    @overrides
    def _score(self, pair: Dict[Text, Any]) -> np.ndarray:
        
        # extract required fields
        inputs = {
            "premise": pair["body"]["premise"],
            "hypothesis": pair["body"]["hypothesis"]
        }
        inputs = self._template.get_prompt_template(**inputs) + self._template.get_completion_template(is_completion=True)
        result = self._pipeline(inputs)
        
        return np.array(result[0]["score"], dtype=np.float32)