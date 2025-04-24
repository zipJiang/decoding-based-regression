"""Run a structure based evaluation.
"""

from overrides import overrides
import evaluate
from tasker import BaseTask
from accelerate.utils import gather_object
from accelerate import PartialState
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
from peft.peft_model import PeftModelForCausalLM
from ..schema_workers.schema import Schema
from ..schema_workers import (
    SchemaOutcome,
    MaieuticSchemaWorker,
    BirdSchemaWorker
)
from ..prob_scorers import (
    DecoderProbScorer,
    EncoderProbScorer
)
from ..chat_templates import (
    UNLITemplate
)
from ..pipelines.level_to_score_pipeline import LevelToScorePipeline


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


@BaseTask.register("structural-evaluation")
class StructuralEvaluationTask(BaseTask):
    __VERSION__ = "0.0.5"
    
    __DATA_PATH_MAP__ = {
        "maieutic-com2sense": "data/graph/maieutic-prompting/Com2Sense.jsonl",
        "maieutic-creak": "data/graph/maieutic-prompting/CREAK.jsonl",
        "maieutic-csqa2": "data/graph/maieutic-prompting/CSQA2.jsonl",
        "bird-com2sense": "data/graph/bird/com2sense-decomposed-sampled.jsonl",
        "bird-today": "data/graph/bird/today-decomposed-sampled.jsonl",
    }

    def __init__(
        self,
        num_labels: int,
        tasks: List[Text],
        output_dir: Text,
        input_dir: Optional[Text] = None
    ):
        super().__init__(output_dir=output_dir)
        self._input_dir = input_dir
        self._partial_state = PartialState()
        # print(self._partial_state)
        self._num_labels = num_labels
        self._tasks = tasks
        
        if self._input_dir is not None:
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
            _config = PeftConfig.from_pretrained(best_ckpt)
            _model = AutoModelForCausalLM.from_pretrained(
                _config.base_model_name_or_path,
                torch_dtype=torch.bfloat16,
                load_in_8bit=False,
                attn_implementation="flash_attention_2",
            )
            _peft_model = PeftModel.from_pretrained(_model, best_ckpt, torch_dtype=torch.bfloat16)
            _tokenizer = AutoTokenizer.from_pretrained(best_ckpt)
            PIPELINE_REGISTRY.register_pipeline(
                "level-to-score",
                pipeline_class=LevelToScorePipeline,
                pt_model=PeftModelForCausalLM,
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
                # minimum -2e8
                selective_logits = torch.where(selective_logits < -2e8, torch.tensor(-2e8, device=selective_logits.device), selective_logits)
                # print(selective_logits)
                step_size = 1 / self._num_labels
                expectation = torch.tensor([[i * step_size + 1 / 2 * step_size for i in range(self._num_labels)]], device=selective_logits.device)
                scores = torch.softmax(selective_logits, dim=-1) @ expectation.T
                scores = scores.squeeze(-1).tolist()
                # indices = torch.argmax(selective_logits, dim=-1).tolist()
                # scores = [i * 0.1 + 0.05 for i in indices]
                # print(scores)
                return scores, selective_logits.tolist()
            
            _pipe = pipeline(
                # "text-generation",
                "level-to-score",
                model=_peft_model,
                max_new_tokens=2,
                tokenizer=_tokenizer,
                device=self._partial_state.device.index,
                level_to_score_func=_level_to_score_func
            )
            
            self._prob_scorer = DecoderProbScorer(
                template=UNLITemplate(),
                pipeline=_pipe
            )
        else:
            self._prob_scorer = EncoderProbScorer(model_name="Zhengping/roberta-large-unli")
        
    @overrides
    def _run(self):
        
        task_results = {}
        
        for taskname in self._tasks:
            
            def _accuracy(
                outcomes: List[SchemaOutcome],
                answer_field: Text = "answer"
            ) -> float:
                """
                """
                correct_outcomes = [outcome for outcome in outcomes if outcome.answer_label == outcome.schema.metadata[answer_field]]
                return len(correct_outcomes) / len(outcomes)
            
            # def _today_accuracy(
            #     schemas_outcomes: List[SchemaOutcome],
            #     modified_schemas_outcomes: List[SchemaOutcome]
            # ) -> float:
            #     """ """
            #     return len([sot for sot, mot in zip(
            #         schemas_outcomes,
            #         modified_schemas_outcomes
            #     ) if sot.score[0] < mot.score[0]]) / len(schemas_outcomes)
            
            if taskname.startswith("maieutic"):
                schema_worker = MaieuticSchemaWorker(
                    prob_scorer=self._prob_scorer,
                    premise_field="rewrite::premise",
                    hypothesis_field="rewrite::hypothesis",
                )
                with open(self.__DATA_PATH_MAP__[taskname], 'r') as file_:
                    data = [json.loads(line.strip()) for line in file_]
                    
                schemas = [Schema.from_dict(item) for item in tqdm(data)]
                conversion_map = {
                    "True": 0,
                    True: 0,
                    "False": 1,
                    False: 1
                }
                for schema, item in zip(schemas, data):
                    schema.metadata["answer"] = conversion_map[schema.metadata["answer"]]

                outcomes = schema_worker(schemas, partial_state=self._partial_state)
                
                task_results[taskname] = {
                    "accuracy": _accuracy(outcomes),
                    "outcomes": [ot.to_dict() for ot in outcomes]
                }
            
            elif taskname.startswith("bird"):
                schema_worker = BirdSchemaWorker(
                    prob_scorer=self._prob_scorer
                )
                
                with open(self.__DATA_PATH_MAP__[taskname], 'r') as file_:
                    data = [json.loads(line.strip()) for line in file_]
                    
                # if taskname == "bird-today":
                #     schemas = [Schema.from_dict(item['schema']) for item in tqdm(data)]
                #     for schema, item in zip(schemas, data):
                #         schema.metadata["answer"] = item["ground_truth"]
                #     schemas_outcomes = schema_worker()
                #     modified_schemas_outcomes = schema_worker([Schema.from_dict(item['modified']) for item in tqdm(data)])
                    
                #     task_results[taskname] = {
                #         "accuracy": _today_accuracy(schemas_outcomes, modified_schemas_outcomes),
                #         "outcomes": [
                #             {
                #                 "original": sot.to_dict(),
                #                 "modified": msot.to_dict()
                #             }
                #             for sot, msot in zip(schemas_outcomes, modified_schemas_outcomes)
                #         ]
                #     }
                
                # elif taskname == "bird-com2sense":
                schemas = [Schema.from_dict(item['schema']) for item in tqdm(data)]
                for schema, item in zip(schemas, data):
                    schema.metadata["answer"] = item["ground_truth"]
                    
                outcomes = schema_worker(schemas, partial_state=self._partial_state)
                
                task_results[taskname] = {
                    "accuracy": _accuracy(outcomes),
                    "outcomes": [ot.to_dict() for ot in outcomes]
                }
            
        return task_results
    
    @overrides
    def _write(self, outputs):
        
        # self._partial_state.on_main_process():
        @self._partial_state.on_main_process
        def _r():
            for taskname, outcomes in outputs.items():
                with open(os.path.join(self._output_dir, f"{taskname}.jsonl"), 'w') as file_:
                    json.dump(outcomes, file_, indent=4)

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
        