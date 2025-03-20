"""
"""

import numpy as np
import datasets
from typing import (
    Text, Union,
    List, Iterable
)
try:
    import ujson as json
except ImportError:
    import json
from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm
from sklearn.isotonic import IsotonicRegression
import os
from overrides import overrides
from .base_dataset_processor import BaseDatasetProcessor
from ..chat_templates import UNLITemplate, BaseTemplate
from ..utils.transforms import _inverse_sigmoid_unli


@BaseDatasetProcessor.register("pseudo-label")
class PseudoLabeledDatasetProcessor(BaseDatasetProcessor):
    def __init__(
        self,
        number_of_levels: int,
        template: BaseTemplate,
        data_dir: Text,
        model_names: List[Text],
        dataset_names: List[Text],
        include_other_tests: bool = False,
        with_system_messages: bool = False,
    ):
        """ """
        super().__init__(template=template)
        self._data_dir = data_dir
        self._template = template
        self._number_of_levels = number_of_levels
        # Test are added to dev
        self._include_other_tests = include_other_tests
        self._with_system_messages = with_system_messages
        self._model_names = [m.split('/')[-1] for m in model_names]
        self._dataset_names = dataset_names

        assert "unli" not in self._dataset_names, "UNLI dataset should not be in the dataset names."
        
    @overrides
    def _process(self) -> Union[Dataset, DatasetDict]:
        """ """
        
        train_data_instance_map = {}
        dev_data_instance_map = {}
        
        unli_dev = datasets.load_dataset("Zhengping/UNLI", split='validation')
        unli_test = datasets.load_dataset("Zhengping/UNLI", split='test')
        unli_train = datasets.load_dataset("Zhengping/UNLI", split='train')
        
        for idx, item in tqdm(enumerate(unli_train), desc="Processing UNLI Train"):
            identifier = ("unli", idx, "train", None)
            # if identifier in train_data_instance_map:
            #     train_data_instance_map[identifier]["label"] = item['label']
            if identifier not in train_data_instance_map:
                train_data_instance_map[identifier] = {
                    "premise": item['premise'],
                    "hypothesis": item['hypothesis'],
                    "pscores": [item['label']]
                }
                
            else:
                train_data_instance_map[identifier]["pscores"].append(item['label'])
                
        for idx, item in tqdm(enumerate(unli_dev), desc="Processing UNLI Test"):
            identifier = ("unli", idx, "test", None)
            # if identifier in test_data_instance_map:
            #     test_data_instance_map[identifier]["label"] = item['label']
            if identifier not in dev_data_instance_map:
                dev_data_instance_map[identifier] = {
                    "premise": item['premise'],
                    "hypothesis": item['hypothesis'],
                    "pscores": [item['label']]
                }
                
            else:
                dev_data_instance_map[identifier]["pscores"].append(item['label'])
                
        def _fix_proba(ist, items):
            # for item in items:
            #     item['processed::probability'] = ist.predict(np.array([item['processed::probability']]).reshape(-1, 1))[0]
                
            # return items
            # first find indices where the probabilities are None
            none_indices = {idx for idx, item in enumerate(items) if item['processed::probability'] is None}
            proba = [item['processed::probability'] if item['processed::probability'] is not None else 0 for item in items]
            proba = np.clip(
                ist.predict(np.array(proba).reshape(-1, 1)),
                a_min=0,
                a_max=10000
            )
            
            for idx, item in enumerate(items):
                if idx in none_indices:
                    item['processed::probability'] = None
                else:
                    item['processed::probability'] = proba[idx]

            return items
        
        for model_name in self._model_names:
            
            # train the isotonic regression model
            with open(os.path.join(self._data_dir, model_name, "unli-00000000.jsonl"), 'r', encoding='utf-8') as file_:
                items = [json.loads(line) for line in file_]

            dev = list(filter(
                lambda x: x['processed::probability'] is not None,
                [item for item in items if item['split'] == 'validation']
            ))
            
            inputs = np.array([d['processed::probability'] for d in dev], dtype=np.float32).reshape(-1, 1)
            # Here we try to use the mapped back values 
            scores = _inverse_sigmoid_unli(np.array([d['label'] for d in dev], dtype=np.float32)) / 10000
            
            ist = IsotonicRegression()
            ist.fit(inputs, scores)
            
            for dataset_name in self._dataset_names:
                shard_index = 0
                while data_path := os.path.join(self._data_dir, model_name, f"{dataset_name}-{shard_index:08d}.jsonl"):
                    if not os.path.exists(data_path):
                        break
                    
                    with open(data_path, 'r', encoding='utf-8') as file_:
                        items = _fix_proba(
                            ist,
                            [json.loads(line) for line in tqdm(file_, desc=f"Processing {model_name} {dataset_name}")]
                        )
                            
                        for item in items:
                            identifier = (item['dataset'], item['idx'], item['split'], item.get("subset", None))
                            if item['split'].startswith("train"):
                                if identifier not in train_data_instance_map:
                                    train_data_instance_map[identifier] = {
                                        "premise": item['premise'],
                                        "hypothesis": item['hypothesis'],
                                        "pscores": [item['processed::probability']]
                                    }
                                else:
                                    train_data_instance_map[identifier]["pscores"].append(item['processed::probability'])
                                    
                            elif self._include_other_tests and item['split'].startswith("test"):
                                if identifier not in dev_data_instance_map:
                                    dev_data_instance_map[identifier] = {
                                        "premise": item['premise'],
                                        "hypothesis": item['hypothesis'],
                                        "pscores": [item['processed::probability']]
                                    }
                                else:
                                    dev_data_instance_map[identifier]["pscores"].append(item['processed::probability'])
                                    
                    shard_index += 1
                    
        # first use dev to create the isotonic regression
        # with open(os.path.join(data_path, self._model_names[0], "unli-00000000.jsonl"), 'r', encoding='utf-8') as file_:
        #     items = [json.loads(line) for line in file_]
        
        sorted_train = sorted(train_data_instance_map.items(), key=lambda x: x[0], reverse=False)
        sorted_dev = sorted(dev_data_instance_map.items(), key=lambda x: x[0], reverse=False)
        
        return DatasetDict({
            "train": self._create_dataset([item for _, item in sorted_train]),
            "validation": self._create_dataset([item for _, item in sorted_dev]),
            "test": self._create_dataset(unli_test.map(lambda x: {"pscores": [x["label"]]}))
        })
        
    def _create_dataset(self, data: Iterable[dict]) -> Dataset:
        """ """
        
        return Dataset.from_list([
            {
                # "premise": item['premise'],
                # "hypothesis": item['hypothesis'],
                "prompt": self._template.get_prompt_template(
                    premise=item['premise'],
                    hypothesis=item['hypothesis']
                ),
                "completion": self._template.get_completion_template(
                    answer=f" <|label_level_0|>"  # Because this label is always there
                ),
                # "scores": _to_dist(item['pscores']),
                "scores": item['pscores'],
            } for item in data
        ])