""" """

from abc import ABC, abstractmethod
from overrides import overrides
import torch
from registrable import Registrable
from dataclasses import dataclass
from typing import List, Dict


class SingleTokenRegLoss(torch.nn.Module, ABC, Registrable):
    """ """
    
    def __init__(
        self,
        scale_factor: float = 1.0,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._scale_factor = scale_factor
    
    def forward(
        self,
        outputs,
        labels,
        scores,
        num_items_in_batch
    ) -> torch.Tensor:
        """We use the scores to calculate the final loss.
        
        logits: [batch_size, seq_len, vocab_size]
        labels: [batch_size, seq_len]
        scores: [batch_size]
        
        We calculate score-loss based on the first token to be decoded.
        """
        logits = outputs['logits'] if isinstance(outputs, dict) else outputs[0]
        
        batch_size, _ = labels.shape
        first_non_ignore_indices = []

        mask = (labels != self._ignore_index)
        # We use the trick that max will return the first index of the max value
        first_non_ignore_indices = torch.argmax(mask.int(), dim=1)
        
        # TODO: Need to handle exception if no non-ignore index is found (mask[x] = 0)
        ...
        
        selected_logits = logits[torch.arange(batch_size), first_non_ignore_indices]  # [batch_size, vocab_size]
        loss = self._compute_loss(selected_logits, scores, num_items_in_batch)
        
        return self._scale_factor * loss
        
    @abstractmethod
    def _compute_loss(
        self,
        selected_logits,
        scores,
        num_items_in_batch
    ):
        """ """
        raise NotImplementedError("Subclasses must implement this method.")
    
    
@SingleTokenRegLoss.register("mse")
class MSESingleTokenRegLoss(SingleTokenRegLoss):
    """ """
    def __init__(
        self,
        rank_dict: dict,
        reduction: str = 'mean',
        *args,
        **kwargs
    ):
        """Giving a set of rank ids, this loss will penalize
        the model through learning-to-rank.
        
        rank_ids: [int]
        """
        super().__init__(*args, **kwargs)
        self._ignore_index = -100
        self._rank_ids = []
        self._rank_targ = []
        
        self._reduction = reduction

        # sort rank_dict by key small to large
        rank_dict = dict(sorted(rank_dict.items(), key=lambda item: item[0]))

        for rank_id, rank_targ in rank_dict.items():
            self._rank_ids.append(rank_id)
            self._rank_targ.append(rank_targ)
        
        self._selection_tensor = torch.nn.Parameter(
            torch.tensor(self._rank_ids, dtype=torch.int64), requires_grad=False
        )
        
        # TODO: Making the data-type configurable
        self._target_tensor = torch.nn.Parameter(
            torch.tensor(self._rank_targ, dtype=torch.bfloat16).unsqueeze(1), requires_grad=False
        )

    @overrides
    def _compute_loss(
        self,
        selected_logits,
        scores,
        num_items_in_batch
    ):
        """ """
        
        pred_scores = torch.matmul(
            torch.nn.functional.softmax(selected_logits[..., self._selection_tensor], dim=-1),
            self._target_tensor
        ).squeeze(-1)  # [batch_size]
        
        return torch.nn.functional.mse_loss(pred_scores, scores, reduction=self._reduction)
    
    
@SingleTokenRegLoss.register("margin")
class MarginSingleTokenRegLoss(SingleTokenRegLoss):
    """ """
    def __init__(
        self,
        rank_dict: dict,
        margin: float = 0.1,
        scale_factor: float = 1.0,
        reduction: str = 'mean',
        *args,
        **kwargs
    ):
        """Giving a set of rank ids, this loss will penalize
        the model through learning-to-rank.
        
        rank_ids: [int]
        """
        super().__init__(*args, **kwargs)
        self._ignore_index = -100
        self._rank_ids = []
        self._rank_targ = []
        
        self._reduction = reduction
        self._margin = margin
        self._scale_factor = scale_factor

        # sort rank_dict by key small to large
        rank_dict = dict(sorted(rank_dict.items(), key=lambda item: item[0]))

        for rank_id, rank_targ in rank_dict.items():
            self._rank_ids.append(rank_id)
            self._rank_targ.append(rank_targ)
        
        self._selection_tensor = torch.nn.Parameter(
            torch.tensor(self._rank_ids, dtype=torch.int64), requires_grad=False
        )
        
        # TODO: Making the data-type configurable
        self._target_tensor = torch.nn.Parameter(
            torch.tensor(self._rank_targ, dtype=torch.bfloat16).unsqueeze(1), requires_grad=False
        )

    @overrides
    def _compute_loss(
        self,
        selected_logits,
        scores,
        num_items_in_batch
    ):
        """ """

        pred_scores = torch.matmul(
            torch.nn.functional.softmax(selected_logits[..., self._selection_tensor], dim=-1),
            self._target_tensor
        ).squeeze(-1)
        
        # Compute pairwise ranking loss
        pos_scores = pred_scores.unsqueeze(1)  # [batch_size, 1]
        neg_scores = pred_scores.unsqueeze(0)  # [1, batch_size]

        pairwise_diff = pos_scores - neg_scores  # [batch_size, batch_size]
        pairwise_gold_diff = scores.unsqueeze(1) - scores.unsqueeze(0)  # [batch_size, batch_size]
        binarized_gold_diff = torch.sign(pairwise_gold_diff)
        
        margin_rank_loss = torch.nn.functional.relu(self._margin - pairwise_diff * binarized_gold_diff)
        
        if self._reduction == 'mean':
            return margin_rank_loss.mean()
        elif self._reduction == 'sum':
            return margin_rank_loss.sum()
        else:
            return margin_rank_loss