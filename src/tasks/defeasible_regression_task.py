"""
"""

import os
import torch
from functools import partial
import uuid
import numpy as np
from overrides import overrides
from datasets import load_from_disk
from accelerate import PartialState
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig
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
from ..rank_dicts import BaseRankDict, SingleLabelRankDict
from ..trainers import DecoderBasedRegressionTrainer
from ..utils.transforms import _discretize_gaussian
from ..data_collators import (
    DataCollatorForCompletionRegression,
    DataCollatorForSingleTokenSoftLM,
    DataCollatorForDefeasibleSoftLM
)
from ..losses import (
    SoftTokenLoss,
    SoftTokenLossWithDefeasibleLoss
)
from ..utils.common import get_tokenizer
import psutil


@BaseTask.register('defeasible-regression')
class TrainDefeasibleRegressionTask(BaseTask):
    __VERSION__ = '0.0.1'

    def __init__(
        self,
        input_dir: Text,
        output_dir: Text,
        learning_rate: float,
        model_name: Text,
        margin: float,
        scale_factor: float,
        # rank_dict: Optional[Lazy[BaseRankDict]] = None,
        # score_loss_func: Optional[Lazy[SingleTokenRegLoss]] = None,
        label_smoothing_factor: Optional[float] = 0.0,
        loss_temperature: Optional[float] = 1.0,
        reverse_kl_loss: Optional[bool] = False,
        std: float = 0.05,
        # force_diffuse: bool = False,
        # is_chat: bool = False
    ):
        super().__init__(output_dir=output_dir)
        self._input_dir = input_dir
        # self._rank_dict = rank_dict.construct(tokenizer=self._tokenizer)
        self._learning_rate = learning_rate
        self._loss_temperature = loss_temperature
        self._reverse_kl_loss = reverse_kl_loss
        self._model_name = model_name
        self._margin = margin
        self._scale_factor = scale_factor
        # self._is_chat = is_chat
        
        # these hyperparameters are used to control the target dist
        self._label_smoothing_factor = label_smoothing_factor
        self._std = std
        
        self._partial_state = PartialState()
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
        
        # Print GPU memory usage statistics
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"GPU Memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        print(f"GPU Max memory allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
        
        self._tokenizer = get_tokenizer(self._model_name)
        self._rank_dict = SingleLabelRankDict.from_tokenizer(self._tokenizer)
        tuples = sorted(self._rank_dict.get_rank_dict(self._tokenizer).items(), key=lambda x: x[1], reverse=False)
        self.levels = np.array([t[1] for t in tuples])

        def _score_map(example) -> Dict[Text, Any]:
            """Unlike the score_map in the sft-training,
            we also transform the `alternate_scores`
            """
            
            # print('-' * 100)
            # print(example['scores'])
            
            number_of_levels = len(self._rank_dict)
            scores = example['scores']
            confidences = example['confidence']
            num_scores = len(scores)
            alternate_scores = example['alternate_scores']
            alternate_confidences = example['alternate_confidence']
            num_alternate_scores = len(alternate_scores)
            if not isinstance(scores, list):
                scores = [scores]
            if not isinstance(alternate_scores, list):
                alternate_scores = [alternate_scores]
            
            # binning = lambda x: max(
            #     min(
            #         int(x * self._number_of_levels / 10000), self._number_of_levels - 1
            #     ), 0
            # )
            
            filtered_scores = [s for s in scores if s is not None and s >= 0] 
            filtered_confidences = np.array([c for s, c in zip(scores, confidences) if s is not None and s >= 0]) ** 2 + 1e-6
            filtered_alternate_scores = [s for s in alternate_scores if s is not None and s >= 0]
            filtered_alternate_confidences = np.array([c for s, c in zip(alternate_scores, alternate_confidences) if s is not None and s >= 0]) ** 2 + 1e-6
            
            if len(filtered_scores) == 0:
                scores = np.ones((self._number_of_levels,), dtype=np.float32) / self._number_of_levels
            else:
                scores = _discretize_gaussian(
                    mean=np.array(filtered_scores, dtype=np.float32),
                    std=self._std,
                    levels=self.levels[np.newaxis, :]
                ) + 1e-6
                
                filtered_confidences = np.expand_dims((filtered_confidences / np.sum(filtered_confidences)), axis=1)
                scores = np.sum(scores * filtered_confidences, axis=0) * len(filtered_scores) + 0.1 / number_of_levels * (num_scores - len(filtered_scores))
                scores = (scores / np.sum(scores)).tolist()
                
            if len(filtered_alternate_scores) == 0:
                alternate_scores = np.ones((self._number_of_levels,), dtype=np.float32) / self._number_of_levels
            else:
                alternate_scores = _discretize_gaussian(
                    mean=np.array(filtered_alternate_scores, dtype=np.float32),
                    std=self._std,
                    levels=self.levels[np.newaxis, :]
                ) + 1e-6
                
                filtered_alternate_confidences = np.expand_dims((filtered_alternate_confidences / np.sum(filtered_alternate_confidences)), axis=1)
                alternate_scores = np.sum(alternate_scores * filtered_alternate_confidences, axis=0) * len(filtered_alternate_scores) + 0.1 / number_of_levels * (num_alternate_scores - len(filtered_alternate_scores))
                alternate_scores = (alternate_scores / np.sum(alternate_scores)).tolist()
            
            return {
                "scores": scores,
                "alternate_scores": alternate_scores,
            }
            
        self._compute_loss_func = SoftTokenLossWithDefeasibleLoss(
            temperature=self._loss_temperature,
            reverse_kl_loss=self._reverse_kl_loss,
            rank_dict=self._rank_dict.get_rank_dict(self._tokenizer),
            margin=self._margin,
            score_loss_scale=self._scale_factor,
        ).to(self._partial_state.device)
            
        with self._partial_state.main_process_first():
            self._train_dataset = load_from_disk(os.path.join(self._input_dir, "dataset", 'train'))
            # self._train_dataset = self._train_dataset.to_iterable_dataset(num_shards=128)
            self._eval_dataset = load_from_disk(os.path.join(self._input_dir, "dataset", 'validation'))
            # self._eval_dataset = self._eval_dataset.to_iterable_dataset(num_shards=16)
        
        self._partial_state.wait_for_everyone()
        
        # convert the score to the target dist
        
        with self._partial_state.main_process_first():
            self._train_dataset = self._train_dataset.map(
                _score_map,
                batched=False,
                # cache_file_name="data/.cache/defeasible_train_scored.arrow",
                # writer_batch_size=100,
                remove_columns=['confidence', 'alternate_confidence']
            )
            self._eval_dataset = self._eval_dataset.map(
                _score_map,
                batched=False,
                # cache_file_name="data/.cache/defeasible_eval_scored.arrow",
                # writer_batch_size=100,
                remove_columns=['confidence', 'alternate_confidence']
            )
            
        self._partial_state.wait_for_everyone()
        
        with self._partial_state.main_process_first():
            self._train_dataset = self._train_dataset.map(
                lambda x: {
                    "messages": x['prompt'] + x['completion'],
                    "alternate_messages": x['alternate_prompt'] + x['alternate_completion'],
                }, remove_columns=['prompt', 'completion', 'alternate_prompt', 'alternate_completion'],
                # cache_file_name="data/.cache/defeasible_train_converted.arrow",
                # writer_batch_size=100,
            )
            self._eval_dataset = self._eval_dataset.map(
                lambda x: {
                    "messages": x['prompt'] + x['completion'],
                    "alternate_messages": x['alternate_prompt'] + x['alternate_completion'],
                }, remove_columns=['prompt', 'completion', 'alternate_prompt', 'alternate_completion'],
                # cache_file_name="data/.cache/defeasible_eval_converted.arrow",
                # writer_batch_size=100,
            )
        
        self._partial_state.wait_for_everyone()
        
        # since we need to prepare two different entries into input_ids,
        # we unfortunately need to process the dataset here.

        def _tokenize(x_batch, prefix: Text = ""):
            processed = self._tokenizer.apply_chat_template(
                x_batch[f'{prefix}messages'],
                tokenize=False,
                tools=None,
            )
            tokenized = self._tokenizer(
                # x_batch['messages'],
                processed,
                add_special_tokens=True,
                padding=False,
                truncation=True,
                max_length=256,
            )
            
            return {
                f"{prefix}input_ids": tokenized['input_ids'],
                f"{prefix}attention_mask": tokenized['attention_mask'],
            }
            
        with self._partial_state.main_process_first():
            self._train_dataset = self._train_dataset.map(
                _tokenize,
                batched=True,
                remove_columns=['messages'],
                # cache_file_name="data/.cache/defeasible_train_tokenized.arrow",
                # writer_batch_size=100,
            ).map(
                partial(_tokenize, prefix="alternate_"),
                batched=True,
                remove_columns=['alternate_messages'],
                # cache_file_name="data/.cache/defeasible_train_alternate_tokenized.arrow",
                # writer_batch_size=100,
            )
            
            self._eval_dataset = self._eval_dataset.map(
                _tokenize,
                batched=True,
                remove_columns=['messages'],
                # cache_file_name="data/.cache/defeasible_eval_tokenized.arrow",
                # writer_batch_size=100,
            ).map(
                partial(_tokenize, prefix="alternate_"),
                batched=True,
                remove_columns=['alternate_messages'],
                # cache_file_name="data/.cache/defeasible_eval_alternate_tokenized.arrow",
                # writer_batch_size=100,
            )
            
        self._partial_state.wait_for_everyone()
        
        self._trainer = DecoderBasedRegressionTrainer(
            model=self._model,
            data_collator=DataCollatorForDefeasibleSoftLM(
                label_smoothing_factor=self._label_smoothing_factor,
                instruction_template="### Question:",
                response_template="### Answer:",
                tokenizer=self._tokenizer,
                # Use default sigma
            ),
            # accelerator_config="configs/accelerate/config.yaml",
            formatting_func=None,
            processing_class=self._tokenizer,
            train_dataset=self._train_dataset,
            eval_dataset=self._eval_dataset,
            args=SFTConfig(
                # accelerator_config="/home/zjiang31/.cache/huggingface/accelerate/default_config.yaml",
                # deepspeed="configs/accelerate/ds_config.json",
                # per_device_train_batch_size=16,
                per_device_train_batch_size=4,
                # gradient_accumulation_steps=2,
                bf16=True,
                dataloader_num_workers=4,
                max_seq_length=256,
                metric_for_best_model="eval_loss",
                learning_rate=self._learning_rate,
                num_train_epochs=1,
                eval_strategy="epoch",
                output_dir=self._output_dir,
                save_total_limit=1,
                save_strategy="epoch",
                report_to="wandb",
                warmup_steps=5000,
                run_name=f"sft_regression::model={self._model_name}::" + str(uuid.uuid4()),
                label_names=["scores"],
                remove_unused_columns=False,
                # eval_on_start=True,
                # lr_scheduler_type="constant",
            ),
            peft_config=self._peft_config,
            # compute_score_loss_func=self._score_loss_func,
            compute_loss_func=self._compute_loss_func,
        )
        
    @overrides
    def _run(self):
        os.environ["WANDB_PROJECT"] = "decoding-based-regression"
        self._trainer.train()
        return self._trainer
        
    @overrides
    def _write(self, outputs):
        ...
        
    @overrides
    def _clean_up(self) -> None:
        """ iteratively remove all the files in the output directory """
        
        @self._partial_state.on_main_process
        def remove_dir(directory: Text) -> None:
            if not os.path.exists(directory):
                return
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                if os.path.isdir(file_path):
                    remove_dir(file_path)
                else:
                    os.remove(file_path)
            os.rmdir(directory)
            
        remove_dir(self._output_dir)
        