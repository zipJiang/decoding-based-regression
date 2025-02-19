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
    Optional,
    List,
    Dict,
    Text,
    Tuple,
    Callable
)
from registrable import Lazy
from ..rank_dicts import BaseRankDict
from ..trainers import DecoderBasedRegressionTrainer
from ..data_collators import DataCollatorForCompletionRegression
from ..losses import SingleTokenRegLoss
from ..utils.common import get_tokenizer


@BaseTask.register('sft-regression')
class TrainSFTRegressionTask(BaseTask):
    __VERSION__ = '0.0.2'

    def __init__(
        self,
        input_dir: Text,
        output_dir: Text,
        learning_rate: float,
        model_name: Text,
        rank_dict: Optional[Lazy[BaseRankDict]] = None,
        score_loss_func: Optional[Lazy[SingleTokenRegLoss]] = None,
        is_chat: bool = False
    ):
        super().__init__(output_dir=output_dir)
        self._input_dir = input_dir
        self._learning_rate = learning_rate
        self._model_name = model_name
        self._is_chat = is_chat
        self._partial_state = PartialState()
        
        self._train_dataset = load_from_disk(os.path.join(self._input_dir, "dataset", 'train'))
        self._train_dataset = self._train_dataset.map(lambda x: {"messages": x['prompt'] + x['completion']}, remove_columns=['prompt', 'completion'])
        # self._train_dataset = self._train_dataset.skip(50000)
        self._eval_dataset = load_from_disk(os.path.join(self._input_dir, "dataset", 'validation'))
        self._eval_dataset = self._eval_dataset.map(lambda x: {"messages": x['prompt'] + x['completion']}, remove_columns=['prompt', 'completion'])
        
        # print(self._train_dataset[0])
        
        self._peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            modules_to_save=["model.embed_tokens", "lm_head"],
        )
        
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            load_in_8bit=False,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            device_map={
                "": self._partial_state.process_index
            },
        )
        
        self._tokenizer = get_tokenizer(self._model_name)
        
        self._score_loss_func = None
        if score_loss_func is not None:
            self._rank_dict = rank_dict.construct(tokenizer=self._tokenizer)
            # print(self._rank_dict.get_rank_dict(self._tokenizer))
            self._score_loss_func = score_loss_func.construct(
                rank_dict=self._rank_dict.get_rank_dict(self._tokenizer)
            )
            self._score_loss_func.to(device=self._partial_state.device)
        
        
        self._trainer = DecoderBasedRegressionTrainer(
            model=self._model,
            data_collator=DataCollatorForCompletionRegression(
                instruction_template="### Question:",
                response_template="### Answer:",
                tokenizer=self._tokenizer
            ),
            formatting_func=None if self._is_chat else lambda x: f"{x['prompt']}\n\n{x['completion']}",
            processing_class=self._tokenizer,
            train_dataset=self._train_dataset,
            eval_dataset=self._eval_dataset,
            args=SFTConfig(
                max_seq_length=256,
                metric_for_best_model="eval_loss",
                learning_rate=self._learning_rate,
                num_train_epochs=5,
                eval_strategy="epoch",
                output_dir=self._output_dir,
                save_total_limit=3,
                save_strategy="epoch",
                report_to="wandb",
                warmup_steps=5000,
                run_name=f"sft_regression::model={self._model_name}::" + str(uuid.uuid4()),
                label_names=["scores"],
                remove_unused_columns=False,
                # lr_scheduler_type="constant",
            ),
            peft_config=self._peft_config,
            compute_score_loss_func=self._score_loss_func
        )
        
    @overrides
    def _run(self):
        os.environ["WANDB_PROJECT"] = "decoding-based-regression"
        self._trainer.train()
        return self._trainer
        
    @overrides
    def _write(self, outputs):
        ...