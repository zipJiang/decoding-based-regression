"""
"""

import os
import torch
import uuid
from overrides import overrides
from datasets import load_from_disk
from accelerate import PartialState
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig
from tasker import BaseTask
from typing import (
    Any,
    List,
    Dict,
    Text,
    Tuple,
    Callable
)
from ..utils.common import get_tokenizer


@BaseTask.register('resize-embeddings')
class ResizeEmbeddingsTask(BaseTask):
    """ """
    __VERSION__ = '0.0.1'
    
    def __init__(
        self,
        model_name: Text,
        number_of_levels: int,
        output_dir: Text,
    ):
        """ """
        super().__init__(output_dir=output_dir)
        self._number_of_levels = number_of_levels
        self._model_name = model_name

    def _run(self):
        """ """

        model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            load_in_8bit=False,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            device_map={
                "": 0
            }
        )

        tokenizer = get_tokenizer(self._model_name)

        # add special tokens
        tokenizer.add_tokens(
            [
                f" <|label_level_{i}|>" for i in range(self._number_of_levels)
            ]
        )
        model.resize_token_embeddings(len(tokenizer))

        return (model, tokenizer)
    
    def _write(self, outputs) -> None:
        """ """
        
        # save model and tokenizer locally so that they can both
        # be loaded with from_pretrained
        
        model, tokenizer = outputs
        model.save_pretrained(self._output_dir)
        tokenizer.save_pretrained(self._output_dir)