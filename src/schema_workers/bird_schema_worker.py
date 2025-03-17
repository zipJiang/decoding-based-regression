"""Works on scoring the bird generated schema.
"""

import numpy as np
from overrides import overrides
from typing import Text, Optional, List, Iterable
from pysat.formula import WCNF
from pysat.examples.rc2 import RC2
from .schema import Schema
from ..prob_scorers.base_prob_scorer import BaseProbScorer
from .base_schema_worker import BaseSchemaWorker, SchemaOutcome


@BaseSchemaWorker.register("bird")
class BirdSchemaWorker(BaseSchemaWorker):
    
    def __init__(
        self,
        prob_scorer: BaseProbScorer,
    ):
        """ """
        super().__init__(prob_scorer=prob_scorer)
        
    @overrides
    def _process_schema(self, schema: Schema) -> SchemaOutcome:
        """ """
        
        schema = schema.copy()
        
        node_grouping = {
            "STORY": [],
            "FACTOR": [],
            "OUTCOME": []
        }
        
        for node in schema.nodes:
            group_name = node.name.split("-")[0]
            node_grouping[group_name].append(node)
            
        assert len(node_grouping["STORY"]) == 1, "There should be exactly one story node."
        assert len(node_grouping["OUTCOME"]) == 2, "There should be exactly two outcome node."
        
        prob_requests = []
        
        for factor_node in node_grouping["FACTOR"]:
            for cidx, candidate in enumerate(factor_node.content['candidates']):
                prob_requests.append({
                    "tag": f"STORY=>{factor_node.name}::CANDIDATES-{cidx}",
                    "body": {
                        "premise": node_grouping["STORY"][0].content["content"],
                        "hypothesis": f"The status for factor ``{factor_node.content['content']}'' is ``{candidate['content']}''"
                    }
                })
                
                # also need to calculate the probability of the candidates to outcomes
                for outcome_node in node_grouping["OUTCOME"]:
                    prob_requests.append({
                        "tag": f"{factor_node.name}::CANDIDATES-{cidx}=>{outcome_node.name}",
                        "body": {
                            "premise": (
                                f"Under the scenario that {node_grouping['STORY'][0].content["content"]}. "
                                f"The status for factor ``{factor_node.content['content']}'' is ``{candidate['content']}''"
                            ),
                            "hypothesis": outcome_node.content["content"]
                        }
                    })

        tag_to_request = {
            pbr["tag"]: pbr for pbr in prob_requests
        }
        
        scores = self._prob_scorer(prob_requests)
        for pbr, score in zip(prob_requests, scores):
            pbr["score"] = score.item()

        def _combination(pool: List[List[Text]]) -> Iterable[List[Text]]:
            if len(pool) == 0:
                yield []
            else:
                for i in range(len(pool[0])):
                    for rest in _combination(pool[1:]):
                        yield [pool[0][i]] + rest

        unnormalized = np.array([0.] * len(node_grouping["OUTCOME"]), dtype=np.float32)
            
        for comb in _combination([
            [f"{factor_node.name}::CANDIDATES-{cidx}" for cidx, _ in enumerate(factor_node.content["candidates"])]
            for factor_node in node_grouping["FACTOR"]
        ]):
            p_f_given_c = np.array(1., dtype=np.float32)
            for cand in comb:
                p_f_given_c *= tag_to_request[f"STORY=>{cand}"]["score"]

            # calculate the probability of the outcomes given the factors using Bordley's formula
            p_o_given_f = np.array([1.] * len(node_grouping["OUTCOME"]), dtype=np.float32)
            
            for cand in comb:
                updates = np.array([
                    tag_to_request[f"{cand}=>{outcome_node.name}"]["score"]
                    for outcome_node in node_grouping["OUTCOME"]
                ], dtype=np.float32)
                p_o_given_f *= updates
                
            unnormalized += p_f_given_c * p_o_given_f
            
        schema.metadata["prob_requests"] = prob_requests
            
        return SchemaOutcome(
            schema=schema,
            score=(unnormalized / np.sum(unnormalized)).tolist(),
            answer_label=np.argmax(unnormalized).item()
        )