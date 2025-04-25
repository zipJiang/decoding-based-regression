import os
import asyncio
import numpy as np
from typing import List, Dict, Any
import json
from src.data_synthesis.prompts import ReasoningSummarizer, ProbabilityJudge

class UncertaintyFilter:
    def __init__(self,
                 qwen_path: str,
                 qwq_path: str,
                 ds_path: str,
                 llama_path: str,
                 discrepancy: float = 0.5,
                 confidence: bool = False,
                 base_output_dir="pseudo-labeled",  
                 judge_dir="_judge",                
                ): 
        
        try:
            with open(ds_path, "r") as f:
                self.deepseek_results = [json.loads(line) for line in f]
            with open(qwen_path, "r") as f:
                self.qwen_results = [json.loads(line) for line in f]
            with open(qwq_path, "r") as f:
                self.qwq_results = [json.loads(line) for line in f]
            with open(llama_path, "r") as f:
                self.llama_results = [json.loads(line) for line in f]
        except FileNotFoundError as e:
            print(f"Error reading model predictions: {e}")

        self.data_name = os.path.basename(ds_path).split(".jsonl")[0]
        self.discrepancy = discrepancy
        self.confidence = confidence
        self.base_output_dir = base_output_dir
        self.judge_dir = judge_dir

        self.score_filter_results = self.score_filter()
    
    def score_filter(self) -> List[Dict[str, Any]]:
        """
        Main method to filter unrealistic scores and aggregate probabilities.
        Reads predictions from multiple model files and handles None probabilities.
        Cases with None probabilities are automatically filtered out.
        
        Args:
            discrepancy: Maximum allowed difference between probabilities (default: 0.5)
            
        Returns:
            List of dictionaries containing filtered results
        """
        filtered_results = []
        filtered_out = [] 
        dataset_labelfield_map = {
            "anli": "label",
            "unli": "snli-label",
            "wanli": "gold"
        }
        dataset = self.deepseek_results[0].get("dataset")
        for i, (ds, qwen, qwq, llama) in enumerate(zip(
            self.deepseek_results, 
            self.qwen_results, 
            self.qwq_results, 
            self.llama_results
        )):
            probs = [
                ds.get("processed::probability"),
                qwen.get("processed::probability"),
                qwq.get("processed::probability"),
                llama.get("processed::probability")
            ]
            label = ds.get(dataset_labelfield_map[dataset])
            if dataset in {"anli", "unli"}:
                is_neutral = label == 1
            elif dataset == "wanli":
                is_neutral = str(label).lower() == "neutral"
            else:
                is_neutral = False 
            valid_probs = [p for p in probs if p is not None]
            should_filter = (max(valid_probs) - min(valid_probs) > self.discrepancy) and is_neutral
            
            result = {
                "idx": i,
                "premise": ds["premise"],
                "hypothesis": ds["hypothesis"],
                "probability w/o judge": probs,
                "filtered": should_filter
            }
            
            if should_filter:
                filtered_out.append(result)
            filtered_results.append(result)
        print(f"Number of filtered out results: {len(filtered_out)}")

        filter_path = f"{self.base_output_dir}/_filter_result/{self.data_name}_{self.discrepancy}.jsonl"
        
        os.makedirs(os.path.dirname(filter_path), exist_ok=True)
        
        if filtered_out:
            with open(filter_path, "w", buffering=8192) as f:  # Use buffered writing
                for result in filtered_out:
                    f.write(json.dumps(result) + "\n")
        
        return filtered_results

    def _reasoning_summarize(self) -> str:
        """
        Summarize reasoning process for deepseek and qwq, save the results.
        """
        ds_path = f"{self.base_output_dir}/_summary/{self.data_name}_ds_{self.discrepancy}.jsonl"
        qwq_path = f"{self.base_output_dir}/_summary/{self.data_name}_qwq_{self.discrepancy}.jsonl"
        if os.path.exists(ds_path) and os.path.exists(qwq_path):
            with open(ds_path,"r") as f:
                ds_reasoning_summaries = [json.loads(line) for line in f]
            with open(qwq_path, "r") as f:
                qwq_reasoning_summaries = [json.loads(line) for line in f]
            return ds_reasoning_summaries, qwq_reasoning_summaries
        
        ds_inputs =  [self.deepseek_results[i["idx"]]["processed::messages"] for i in self.score_filter_results if i["filtered"]]
        qwq_inputs = [self.qwq_results[i["idx"]]["processed::messages"] for i in self.score_filter_results if i["filtered"]]
        
        task = ReasoningSummarizer()
        ds_reasoning_summaries = task.get_result(ds_inputs)
        qwq_reasoning_summaries = task.get_result(qwq_inputs)
        filtered_out = [i for i in self.score_filter_results if i["filtered"]]
        ds_results = [
            {
                "idx": i["idx"],
                "premise": i["premise"],
                "hypothesis": i["hypothesis"],
                **ds_sum
            }
            for i, ds_sum in zip(filtered_out, ds_reasoning_summaries) 
        ]

        qwq_results = [
            {
                "idx": i["idx"],
                "premise": i["premise"],
                "hypothesis": i["hypothesis"],
                **qwq_sum
            }
            for i, qwq_sum in zip(filtered_out, qwq_reasoning_summaries)
        ]
        os.makedirs(f"{self.base_output_dir}/_summary", exist_ok=True)
        with open(ds_path, 'w', encoding='utf-8',buffering=8192) as f:
            for result in ds_results:
                f.write(json.dumps(result) + "\n")
        with open(qwq_path, 'w', encoding='utf-8',buffering=8192) as f:
            for result in qwq_results:
                f.write(json.dumps(result) + "\n")

        return ds_reasoning_summaries, qwq_reasoning_summaries

    
    def score_judge(self,) -> List:
        """
        Judge scores from multiple models and filter unrealistic scores.
        """

        judge_path = f"{self.base_output_dir}/{self.judge_dir}/{self.data_name}_{self.discrepancy}.json"
        if os.path.exists(judge_path):
            with open(judge_path, "r") as f:
                # results = [json.loads(line) for line in f]
                results = json.load(f)
            return results

        filtered_results = [i for i in self.score_filter_results if i["filtered"]]  
        ds_reasoning_summaries, qwq_reasoning_summaries = self._reasoning_summarize()
            
        judge = ProbabilityJudge()

        assert len(ds_reasoning_summaries) == len(filtered_results)
        inputs = [
            {       
                "idx": i["idx"],
                "premise": i["premise"],
                "hypothesis": i["hypothesis"],
                "reasoning_1": ds_reasoning_summaries[idx]["reasoning"],
                "reasoning_2": self.qwen_results[i["idx"]]["processed::messages"],
                "reasoning_3": qwq_reasoning_summaries[idx]["reasoning"],
                "reasoning_4": self.llama_results[i["idx"]]["processed::messages"],
            }
            for idx,i in enumerate(filtered_results)
        ]
        if self.confidence:
            results = judge.confidence_judge(inputs)   
        else:
            results = judge.basic_judge(inputs)

        os.makedirs(f"{self.base_output_dir}/{self.judge_dir}", exist_ok=True)
        with open(judge_path, "w",  encoding='utf-8', buffering=8192) as f:
            f.write(json.dumps(results, indent=4))
        return results

    def score_aggregation(self) -> List:
        """
        Aggregate scores from multiple models and filter unrealistic scores.
        Handles both confidence-based and basic judging.
        """
        results = self.score_judge()

        assert len(results) == len([i for i in self.score_filter_results if i["filtered"]])

        outputs = self.deepseek_results.copy()  # Avoid modifying original data
        keys_to_remove = ['processed::reasoning', 'processed::thinking', 'processed::messages', 'processed::probability']

        for output, filtered in zip(outputs, self.score_filter_results):
            for key in keys_to_remove:
                output.pop(key, None)
            output["probability w/o judge"] = filtered["probability w/o judge"]
            # Only set final probability if not using confidence scoring
            if not self.confidence:
                output["probability"] = filtered["probability w/o judge"]

        for result in results:
            index = result["index"]
            if index < len(outputs):
                outputs[index].update(result)

        output_dir = f"{self.base_output_dir}/{self.judge_dir}_aggregate"
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, f"{self.data_name}_{self.discrepancy}.json")
        with open(output_path, "w", buffering=8192) as f:
            f.write(json.dumps(outputs, indent=4))

        return outputs
