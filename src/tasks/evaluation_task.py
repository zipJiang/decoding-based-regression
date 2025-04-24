""" """

from overrides import overrides
import evaluate
import numpy as np
from tasker import BaseTask
from datasets import load_from_disk, load_dataset
# from transformers.trainer_utils import EvalPrediction
from tqdm import tqdm
import os
import uuid
import re
try:
    import ujson as json
except ImportError:
    import json
import torch
import logging
from accelerate import PartialState
from accelerate.utils import gather_object
from typing import (
    Text,
    Dict,
    Optional,
    Tuple,
    List
)
from transformers.pipelines import PIPELINE_REGISTRY
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, PreTrainedTokenizer
from transformers import GenerationConfig
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import PeftModel, PeftConfig
from ..pipelines.level_to_score_pipeline import LevelToScorePipeline


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


@BaseTask.register("evaluation")
class EvaluationTask(BaseTask):
    __VERSION__ = "0.1.0"
    
    __DEFEASIBLE_LIST__ = {
        "defeasible-atomic",
        "defeasible-snli",
    }

    def __init__(
        self,
        input_dir: Text,
        num_labels: int,
        dataset_map: List[Tuple[Text, Text]],
        output_dir: Text
    ):
        super().__init__(output_dir=output_dir)
        self._input_dir = input_dir
        self._num_labels = num_labels
        print(input_dir)
        # self._dataset_dir = dataset_dir
        self._dataset_map = {
            dataset_name: load_from_disk(os.path.join(dataset_dir, "dataset", 'test'))
            for dataset_name, dataset_dir in dataset_map
        }
        
        self._partial_state = PartialState()
        
        def _parse_ckpt_dir(directory: Text) -> int:
            match = re.search(r"checkpoint-(\d+)", directory)
            return int(match.group(1))

        
        ckpts = []
        for ckpt_dir in os.listdir(self._input_dir):
            if not os.path.isdir(os.path.join(self._input_dir, ckpt_dir)):
                continue
            ckpt_num = (ckpt_dir, _parse_ckpt_dir(ckpt_dir))
            ckpts.append(ckpt_num)
            
        # find the latest
        latest_ckpt_dir = sorted(ckpts, key=lambda x: x[1], reverse=True)[0][0]
        logger.info(f"Loading the state file checkpoint from {latest_ckpt_dir}.")
        
        with open(os.path.join(self._input_dir, latest_ckpt_dir, "trainer_state.json") ,'r', encoding='utf-8') as file_:
            training_state = json.load(file_)
            best_ckpt = training_state['best_model_checkpoint']
        logger.info(f"Best checkpoint is {best_ckpt}.")

        # load the best ckpt
        self._config = PeftConfig.from_pretrained(best_ckpt)
        self._model = AutoModelForCausalLM.from_pretrained(
            self._config.base_model_name_or_path,
            torch_dtype=torch.bfloat16,
            load_in_8bit=False,
            attn_implementation="flash_attention_2",
        )
        self._peft_model = PeftModel.from_pretrained(self._model, best_ckpt, torch_dtype=torch.bfloat16)
        self._tokenizer = AutoTokenizer.from_pretrained(best_ckpt)
        
    @overrides
    def _run(self):
        """ """
        
        PIPELINE_REGISTRY.register_pipeline(
            "level-to-score",
            pipeline_class=LevelToScorePipeline,
            pt_model=AutoModelForCausalLM
        )

        def _level_to_score_func(
            logits: Tuple[torch.FloatTensor],
            tokenizer: PreTrainedTokenizer
        ) -> Tuple[List[float], List[float]]:
            """ """
            # TODO: factor the number_of_levels out as configurable
            # parameters
            logits = logits[0]
            considering_ids = tokenizer.convert_tokens_to_ids([f" <|label_level_{i}|>" for i in range(self._num_labels)])
            # print(considering_ids, logits.shape)
            selective_logits = torch.index_select(logits, 1, torch.tensor(considering_ids, device=logits.device))
            # print(selective_logits)
            step_size = 1 / self._num_labels
            expectation = torch.tensor([[i * step_size + 1 / 2 * step_size for i in range(self._num_labels)]], device=selective_logits.device)
            scores = torch.softmax(selective_logits, dim=-1) @ expectation.T
            scores = scores.squeeze(-1).tolist()
            # indices = torch.argmax(selective_logits, dim=-1).tolist()
            # scores = [i * 0.1 + 0.05 for i in indices]
            # print(scores)
            return scores, selective_logits.tolist()
        
        pipe = pipeline(
            # "text-generation",
            "level-to-score",
            model=self._peft_model,
            max_new_tokens=2,
            tokenizer=self._tokenizer,
            device=self._partial_state.device.index,
            level_to_score_func=_level_to_score_func,
            torch_dtype=torch.bfloat16,
        )
        # pipe = pipe.to(self._partial_state.device.index)
        spearman_evaluator = evaluate.load("src/metrics/decorel_regression.py", correlation_type="spearman")
        pearson_evaluator = evaluate.load("src/metrics/decorel_regression.py", correlation_type="pearson")
        accuracy_evaluator = evaluate.load("accuracy")
        
        result_dict = {}
        
        def _distributed_inference(inputs_, pipe_):
            """ """
            with self._partial_state.split_between_processes(inputs_) as distributed_inputs:
                process_id = self._partial_state.process_index
                results = [pipe_(ipt, do_sample=False) for ipt in tqdm(distributed_inputs, desc=f"Process {process_id}", position=process_id, leave=True)]

            self._partial_state.wait_for_everyone()
            results = gather_object(results)
            
            return results
        
        for dataset_name, test_dataset in self._dataset_map.items():
            if dataset_name == "hellaswag":
                # hella_swag is special and needs to be handled differently
                inputs_a = [
                    datapiece['prompt_a'] + [
                        {
                            "role": "assistant",
                            "content": "### Answer:"
                        }
                    ]
                    for datapiece in test_dataset
                ]
                inputs_b = [
                    datapiece['prompt_b'] + [
                        {
                            "role": "assistant",
                            "content": "### Answer:"
                        }
                    ]
                    for datapiece in test_dataset
                ]
                inputs_c = [
                    datapiece['prompt_c'] + [
                        {
                            "role": "assistant",
                            "content": "### Answer:"
                        }
                    ]
                    for datapiece in test_dataset
                ]
                inputs_d = [
                    datapiece['prompt_d'] + [
                        {
                            "role": "assistant",
                            "content": "### Answer:"
                        }
                    ]
                    for datapiece in test_dataset
                ]
                
                # results_a = [
                #     pipe(ipt, do_sample=False)
                #     for ipt in tqdm(inputs_a)
                # ]

                results_a = _distributed_inference(inputs_a, pipe)
                results_b = _distributed_inference(inputs_b, pipe)
                results_c = _distributed_inference(inputs_c, pipe)
                results_d = _distributed_inference(inputs_d, pipe)
                
                # results_b= [
                #     pipe(ipt, do_sample=False)
                #     for ipt in tqdm(inputs_b)
                # ]
                
                # results_c = [
                #     pipe(ipt, do_sample=False)
                #     for ipt in tqdm(inputs_c)
                # ]
                
                # results_d = [
                #     pipe(ipt, do_sample=False)
                #     for ipt in tqdm(inputs_d)
                # ]
                
                if self._partial_state.is_main_process:
                    results = [
                        {
                            "p_a": r_a[0],
                            "p_b": r_b[0],
                            "p_c": r_c[0],
                            "p_d": r_d[0]
                        } 
                        for i, (r_a, r_b, r_c, r_d) in enumerate(zip(results_a, results_b, results_c, results_d))
                    ]
                    
                    predictions = np.argmax(
                        np.array(
                            [[r['p_a']['score'], r['p_b']['score'], r['p_c']['score'], r['p_d']['score']] for r in results]
                        ), axis=1
                    )
                    labels = test_dataset['label']
                    accuracy_evaluator.compute(predictions=predictions, references=labels)
                    
                    result_dict[dataset_name] = {
                        "results": results,
                        "evaluation": {
                            "accuracy": accuracy_evaluator.compute(predictions=predictions, references=labels)
                        }
                    }
                
            elif dataset_name == "copa":
                # Though similar, COPA is different from UNLI, as it replaces inputs instead
                # of appending them
                inputs_a = [
                    datapiece['prompt_a'] + [
                        {
                            "role": "assistant",
                            "content": "### Answer:"
                        }
                    ]
                    for datapiece in test_dataset
                ]
                
                inputs_b = [
                    datapiece['prompt_b'] + [
                        {
                            "role": "assistant",
                            "content": "### Answer:"
                        }
                    ]
                    for datapiece in test_dataset
                ]
                
                # results_a = [
                #     pipe(ipt, do_sample=False)
                #     for ipt in tqdm(inputs_a)
                # ]
                
                # results_b = [
                #     pipe(ipt, do_sample=False)
                #     for ipt in tqdm(inputs_b)
                # ]
                
                results_a = _distributed_inference(inputs_a, pipe)
                results_b = _distributed_inference(inputs_b, pipe)
                
                if self._partial_state.is_main_process:
                    
                    results = [
                        {
                            "p_a": r_a[0],
                            "p_b": r_b[0]
                        }
                        for r_a, r_b in zip(results_a, results_b)
                    ]
                    
                    result_dict[dataset_name] = {
                        "results": results,
                        "evaluation": {
                            "accuracy": accuracy_evaluator.compute(
                                predictions=[int(r['p_b']['score'] - r['p_a']['score'] > 0) for r in results],
                                references=test_dataset['more_likely_index']
                            )
                        }
                    }
                
            else:
                inputs = [
                    datapiece['prompt'] + [
                    # {
                    #     "role": "user",
                    #     "content": datapiece["prompt"]
                    # },
                    {
                        "role": "assistant",
                        # TODO: Switch to the correct template class
                        "content": "### Answer:"
                    }
                ] for datapiece in test_dataset]
            
                # results = [pipe(
                #     ipt,
                #     do_sample=False
                # ) for ipt in tqdm(inputs)]

                results = _distributed_inference(inputs, pipe)
                
                if dataset_name in self.__DEFEASIBLE_LIST__:
                    # This only works for defeasible evaluation
                    update_inputs = [
                        datapiece['update_prompt'] + [{
                            "role": "assistant",
                            # TODO: Switch to the correct template class
                            "content": "### Answer:"
                        }] for datapiece in test_dataset
                    ]
                    # update_results = [pipe(
                    #     ipt,
                    #     do_sample=False
                    # ) for ipt in tqdm(update_inputs)]
                    
                    update_results = _distributed_inference(update_inputs, pipe)
                    
                    if self._partial_state.is_main_process:
                        
                        result_dict[dataset_name] = {
                            "results": [
                                {
                                    "original": r[0],
                                    "update": u[0]
                                } for r, u in zip(results, update_results)
                            ],
                            "evaluation": {
                                # defeasible 0 for weakener, 1 for strengthener
                                "accuracy": accuracy_evaluator.compute(
                                    predictions=[int(u[0]['score'] - r[0]['score'] > 0) for r, u in zip(results, update_results)],
                                    references=test_dataset['is_strengthener']
                                ),
                            }
                        }
                
                # This only works for non-defeasible evaluation
                else:
                    result_dict[dataset_name] = {
                        "results": results,
                        "evaluation": {
                            "spearman": spearman_evaluator.compute(
                                predictions=[r[0]['score'] for r in results],
                                # references=load_dataset("Zhengping/UNLI", split='test')['label']
                                # references=[datapiece['scores'] for datapiece in self._test_dataset][0:3040:30]
                                references=test_dataset['scores']
                            ),
                            "pearson": pearson_evaluator.compute(
                                predictions=[r[0]['score'] for r in results],
                                # references=load_dataset("Zhengping/UNLI", split='test')['label']
                                # references=[datapiece['scores'] for datapiece in self._test_dataset][0:3040:30]
                                references=test_dataset['scores']
                            )
                        }
                    }
            
        return result_dict

    @overrides
    def _write(self, outputs):
        """ """
        # merged_model, tokenizer = outputs
        # merged_model.save_pretrained(self._output_dir)
        # tokenizer.save_pretrained(self._output_dir)
        # with open(os.path.join(self._output_dir, "evaluation.json"), 'w', encoding='utf-8') as file_:
        #     json.dump(outputs, file_, ensure_ascii=False, indent=4)
        
        # we separate the result into different files
        @self._partial_state.on_main_process
        def _r():
            for dataset_name, result in outputs.items():
                with open(os.path.join(self._output_dir, f"{dataset_name}.json"), 'w', encoding='utf-8') as file_:
                    json.dump(result, file_, ensure_ascii=False, indent=4)

        _r()
        
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
        