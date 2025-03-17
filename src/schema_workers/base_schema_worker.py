""" """
import abc
import numpy as np
from dataclasses import dataclass
from registrable import Registrable
from typing import Union, List, Text, Optional
from tqdm import tqdm
from .schema import Schema
from ..prob_scorers.base_prob_scorer import BaseProbScorer


@dataclass
class SchemaOutcome:
    """ """
    # distribution: np.ndarray
    answer_label: int
    score: Optional[float | List[float]] = None
    schema: Optional[Schema] = None
    
    def to_dict(self) -> dict:
        """ """
        return {
            'answer_label': self.answer_label,
            'score': self.score,
            'schema': self.schema.to_dict()
        }


class BaseSchemaWorker(abc.ABC, Registrable):
    """ """
    def __init__(
        self,
        prob_scorer: BaseProbScorer
    ):
        super().__init__()
        self._prob_scorer = prob_scorer
        
    def __call__(self, schema: Union[Schema, List[Schema]]) -> Union[Schema, List[SchemaOutcome]]:
        """ """
        
        if isinstance(schema, list):
            return self._process_schemas(schema)
        
        return self._process_schema(schema)
            
    def _process_schemas(self, schemas: List[Schema]) -> List[SchemaOutcome]:
        """This gives a most standard way to process a list of schemas.
        """
        return [self._process_schema(schema) for schema in tqdm(schemas)]
    
    @abc.abstractmethod
    def _process_schema(self, schema: Schema) -> SchemaOutcome:
        """This gives a most standard way to process a schema.
        """
        raise NotImplementedError()