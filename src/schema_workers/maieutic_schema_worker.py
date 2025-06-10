"""A worker that works on maieutic prompt schemas.
"""

import numpy as np
from overrides import overrides
from typing import Text, Optional
from accelerate import PartialState
from accelerate.utils import gather_object
from pysat.formula import WCNF
from pysat.examples.rc2 import RC2
from .schema import Schema
from ..prob_scorers.base_prob_scorer import BaseProbScorer
from .base_schema_worker import BaseSchemaWorker, SchemaOutcome


@BaseSchemaWorker.register("maieutic")
class MaieuticSchemaWorker(BaseSchemaWorker):
    
    __SCALE__ = 10000
    
    def __init__(
        self,
        prob_scorer: BaseProbScorer,
        premise_field: Optional[Text] = "rewrite::premise",
        hypothesis_field: Optional[Text] = "rewrite::hypothesis",
    ):
        """ """
        super().__init__(prob_scorer=prob_scorer)
        self._premise_field = premise_field
        self._hypothesis_field = hypothesis_field
        
    @overrides
    def _process_schema(
        self,
        schema: Schema,
        partial_state: PartialState | None = None
    ) -> SchemaOutcome:
        """ """
        
        # get the premise and hypothesis
        schema = schema.copy()
        
        prob_requests = []

        for node in schema.nodes:
            if node.content['integrity']:
                prob_requests.append({
                    "tag": node.name,
                    "type": "belief",
                    "body": {
                        "premise": node.content[self._premise_field],
                        "hypothesis": node.content[self._hypothesis_field]
                    }
                })
            elif node.name == "Q":
                prob_requests.append({
                    "tag": node.name,
                    "type": "belief",
                    "body": {
                        "premise": node.content[self._premise_field],
                        "hypothesis": node.content[self._hypothesis_field]
                    }
                })
                
        name_to_node_dict = {
            node.name: node for node in schema.nodes
        }

        for edge in schema.edges:
            prob_requests.append({
                "tag": f"{edge.source} -> {edge.target}",
                "type": "consistency",
                "body": {
                    "premise": name_to_node_dict[edge.target].content["E"],
                    "hypothesis": name_to_node_dict[edge.source].content["E"] if edge.content["direction"] == "T" else name_to_node_dict[edge.target].content["E_tilde"]
                },
                "direction": edge.content['direction']
            })
            
        with partial_state.split_between_processes(prob_requests) as distributed_prob_requests:
            scores = self._prob_scorer(distributed_prob_requests)
            scores = scores.tolist()

        partial_state.wait_for_everyone()
        scores = gather_object(scores)
        scores = np.array(scores, dtype=np.float32).flatten().tolist()
        
        for pbr, score in zip(prob_requests, scores):
            pbr["score"] = score

        # add scores to nodes and edges
        tag_to_struct = {
            **{
                node.name: node for node in schema.nodes
            },
            **{
                f"{edge.source} -> {edge.target}": edge for edge in schema.edges
            }
        }
        
        for pbr in prob_requests:
            tag_to_struct[pbr["tag"]].content["score"] = pbr["score"]
        
        # construct max-SAT problem
        # create conversion table node_name to index
        node_name_to_index = {node.name: i + 1 for i, node in enumerate(schema.nodes)}
        # print(list(node_name_to_index.keys()))
        wcnf = WCNF()
        
        # Add belief constraints as weighted soft clauses
        for pbr in prob_requests:
            if pbr["type"] == "belief":
                clause = [node_name_to_index[pbr["tag"]]]
                wcnf.append(clause, weight=int(pbr["score"] * self.__SCALE__))  # Scale scores for weight
                clause_neg = [-node_name_to_index[pbr["tag"]]]
                wcnf.append(clause_neg, weight=int((1 - pbr["score"]) * self.__SCALE__))  # Scale scores for weight

        # Add consistency constraints
        for pbr in prob_requests:
            if pbr["type"] == "consistency":
                source_index = node_name_to_index[pbr["tag"].split(" -> ")[0]]
                target_index = node_name_to_index[pbr["tag"].split(" -> ")[1]]
                
                if pbr["direction"] == "T":
                    clause = [source_index, -target_index]
                else:
                    clause = [-source_index, -target_index]
                
                wcnf.append(clause, weight=int(pbr["score"] * self.__SCALE__))  # Scale scores for weight

        with RC2(wcnf.copy()) as rc2:
            solution = rc2.compute()

        # print(solution, prob_requests)
        
        # write solution back to schema
        for node in schema.nodes:
            node.content['solution'] = solution[node_name_to_index[node.name] - 1] > 0

        schema.metadata["prob_requests"] = prob_requests

        return SchemaOutcome(
            score=None,
            # answer_label="True" if solution[node_name_to_index['Q'] - 1] > 0 else "False",
            answer_label=0 if solution[node_name_to_index['Q'] - 1] > 0 else 1,
            schema=schema
        )