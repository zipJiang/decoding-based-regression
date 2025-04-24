"""
"""

import evaluate
import numpy as np
from scipy.stats import pearsonr, spearmanr
from evaluate import EvaluationModuleInfo
from datasets import Features, Value

class AverageAbsoluteErrorMetric(evaluate.Metric):
    
    def __init__(
        self,
        correlation_type: str = 'pearson',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.correlation_type = correlation_type
    
    def _info(self):
        return EvaluationModuleInfo(
            description="This metric calculates correlation metrics for decoding-based regression tasks.",
            citation="""@inproceedings{chen-etal-2020-uncertain,
                title = "Uncertain Natural Language Inference",
                author = "Chen, Tongfei  and
                Jiang, Zhengping  and
                Poliak, Adam  and
                Sakaguchi, Keisuke  and
                Van Durme, Benjamin",
                editor = "Jurafsky, Dan  and
                Chai, Joyce  and
                Schluter, Natalie  and
                Tetreault, Joel",
                booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
                month = jul,
                year = "2020",
                address = "Online",
                publisher = "Association for Computational Linguistics",
                url = "https://aclanthology.org/2020.acl-main.774/",
                doi = "10.18653/v1/2020.acl-main.774",
                pages = "8772--8779",
                abstract = "We introduce Uncertain Natural Language Inference (UNLI), a refinement of Natural Language Inference (NLI) that shifts away from categorical labels, targeting instead the direct prediction of subjective probability assessments. We demonstrate the feasibility of collecting annotations for UNLI by relabeling a portion of the SNLI dataset under a probabilistic scale, where items even with the same categorical label differ in how likely people judge them to be true given a premise. We describe a direct scalar regression modeling approach, and find that existing categorically-labeled NLI data can be used in pre-training. Our best models correlate well with humans, demonstrating models are capable of more subtle inferences than the categorical bin assignment employed in current NLI tasks."
            }
            """,
            inputs_description="Predictions and references should be lists or arrays of float values.",
            features=Features({
                'predictions': Value('float32'),
                'references': Value('float32')
            }),
        )

    def _compute(self, predictions, references):
        """ """

        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length.")
        
        predictions = np.array(predictions)
        references = np.array(references)
        
        if self.correlation_type == 'pearson':
            correlation, _ = pearsonr(predictions, references)
        elif self.correlation_type == 'spearman':
            correlation, _ = spearmanr(predictions, references)
            
        return {
            "correlation": correlation
        }