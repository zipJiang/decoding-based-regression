""" """

from overrides import overrides
import evaluate
from tasker import BaseTask
from datasets import load_from_disk, load_dataset
# from transformers.trainer_utils import EvalPrediction
import os
import uuid
import re
try:
    import ujson as json
except ImportError:
    import json
import torch
import logging
from typing import (
    Text,
    Dict,
    Optional,
    Tuple,
    List
)
from transformers.pipelines import PIPELINE_REGISTRY
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, PreTrainedTokenizer
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
    __VERSION__ = "0.0.3"

    def __init__(
        self,
        input_dir: Text,
        dataset_dir: Text,
        output_dir: Text
    ):
        super().__init__(output_dir=output_dir)
        self._input_dir = input_dir
        self._dataset_dir = dataset_dir
        self._test_dataset = load_from_disk(os.path.join(self._dataset_dir, "dataset", 'test'))
        
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
        self._model = AutoModelForCausalLM.from_pretrained(self._config.base_model_name_or_path)
        self._peft_model = PeftModel.from_pretrained(self._model, best_ckpt)
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
        ) -> List[float]:
            """ """
            # TODO: factor the number_of_levels out as configurable
            # parameters
            logits = logits[0]
            considering_ids = tokenizer.convert_tokens_to_ids([f" <|label_level_{i}|>" for i in range(10)])
            # print(considering_ids, logits.shape)
            selective_logits = torch.index_select(logits, 1, torch.tensor(considering_ids).to(logits.device))
            # print(selective_logits)
            expectation = torch.tensor([[i * 0.1 + 0.05 for i in range(10)]])
            scores = torch.softmax(selective_logits, dim=-1) @ expectation.T
            scores = scores.squeeze(-1).tolist()
            # indices = torch.argmax(selective_logits, dim=-1).tolist()
            # scores = [i * 0.1 + 0.05 for i in indices]
            # print(scores)
            return scores
        
        pipe = pipeline(
            # "text-generation",
            "level-to-score",
            model=self._model,
            tokenizer=self._tokenizer,
            device=0,
            level_to_score_func=_level_to_score_func
        )

        inputs = [[
            {
                "role": "user",
                "content": datapiece["prompt"]
            },
            {
                "role": "assistant",
                "content": "### Answer:"
            }
        ] for datapiece in self._test_dataset]
        
        task_evaluator = evaluate.load("src/metrics/decorel_regression.py")
        
        results = pipe(
            inputs,
            do_sample=False
        )
        
        return (
            results,
            task_evaluator.compute(
                predictions=[r[0]['score'] for r in results],
                # references=load_dataset("Zhengping/UNLI", split='test')['label']
                references=self._test_dataset['label']
            )
        )

    @overrides
    def _write(self, outputs):
        """ """
        # merged_model, tokenizer = outputs
        # merged_model.save_pretrained(self._output_dir)
        # tokenizer.save_pretrained(self._output_dir)
        
        results, evaluation = outputs
        
        with open(os.path.join(self._output_dir, "results.json"), 'w', encoding='utf-8') as file_:
            json.dump(results, file_, ensure_ascii=False, indent=4)
            
        with open(os.path.join(self._output_dir, "evaluation.json"), 'w', encoding='utf-8') as file_:
            json.dump(evaluation, file_, ensure_ascii=False, indent=4)