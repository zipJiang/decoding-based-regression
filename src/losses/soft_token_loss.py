"""Different from single_token_reg_loss,
the logits isn't selected.
"""

from abc import ABC, abstractmethod
from overrides import overrides
from .single_token_reg_loss import MarginSingleTokenRegLoss
import torch
from registrable import Registrable
from dataclasses import dataclass
from typing import List, Dict


class SoftTokenLoss(torch.nn.Module):
    """ """

    _ignore_index: int = -100
    
    def __init__(
        self,
        reverse_kl_loss: bool = False,
        temperature: float = 1.0,
    ):
        """ """
        super().__init__()
        self._reverse_kl_loss = reverse_kl_loss
        self._temperature = temperature
        
    def forward(
        self,
        outputs,
        labels,
        num_items_in_batch,
        scores,
        is_defeasible
    ) -> torch.Tensor:
        """ Different from normal token loss,
        we use a soft label [batch_size, seq_len, vocab_size] to calculate the loss.

        the first slice over the second_dim can be used to calculate a ignore_index mask.
        """
        
        logits = outputs['logits'] if isinstance(outputs, dict) else outputs[0]
        # mask = labels[:, :, 0].eq(self._ignore_index)  # [batch_size, 1, vocab_size]
        
        # calculate KL divergence between the soft label and the logits
        return self._compute_loss(logits, labels, num_items_in_batch)
        
    def _compute_loss(
        self,
        logits,
        labels,
        num_items_in_batch
    ) -> torch.Tensor:
        """ """

        # shift the labels to the right
        logits = logits[:, :-1, :].contiguous()
        labels = labels[:, 1:, :].contiguous()
        mask = labels[:, :, 0].eq(self._ignore_index)
        
        labels.masked_fill_(
            labels.eq(self._ignore_index),
            1.0
        )
        
        # pos_loss = torch.nn.functional.cross_entropy(
        #     logits.view(-1, logits.size(-1)),
        #     torch.clamp(labels, min=0).view(-1, labels.size(-1)),
        #     reduction='none'
        # )
        
        # logits: [batch_size, seq_len, vocab_size]
        # labels: [batch_size, seq_len, vocab_size]
        
        # assume that labels are the soft labels
        if not self._reverse_kl_loss:
            pos_loss = torch.nn.functional.kl_div(
                torch.log_softmax(logits / self._temperature, dim=-1),
                labels,
                reduction='none',
            )
        else:
            # reversed kl divergence
            pos_loss = torch.nn.functional.kl_div(
                torch.log(labels),
                torch.log_softmax(logits / self._temperature, dim=-1),
                log_target=True,
                reduction="none",
            )

        # pos_loss: [batch_size, seq_len, vocab_size]
        pos_loss = pos_loss.sum(dim=-1)
        
        # We only calculate the loss for the non-masked tokens
        pos_loss.masked_fill_(mask, 0)
        num_active_items = mask.numel() - mask.sum()

        loss = pos_loss.sum() / num_active_items
        
        return loss
    
    
class SoftTokenLossWithDefeasibleLoss(torch.nn.Module):
    """ """
    
    _ignore_index: int = -100
    
    def __init__(
        self,
        rank_dict: dict,
        margin: float = 0.0,
        score_loss_scale: float = 1.0,
        reverse_kl_loss: bool = False,
        temperature: float = 1.0,
    ):
        """ """
        super().__init__()
        self._rank_dict = dict(sorted(rank_dict.items(), key=lambda item: item[0]))
        self._score_loss_scale = score_loss_scale
        self._reverse_kl_loss = reverse_kl_loss
        self._temperature = temperature
        
        self._rank_ids = []
        self._rank_targ = []
        
        for rank_id, rank_targ in self._rank_dict.items():
            self._rank_ids.append(rank_id)
            self._rank_targ.append(rank_targ)
            
        self._margin = torch.nn.Parameter(
            torch.tensor(margin, dtype=torch.float32),
            requires_grad=False
        )
            
        self._selection_tensor = torch.nn.Parameter(
            torch.tensor(self._rank_ids, dtype=torch.int64), requires_grad=False
        )
        
        # TODO: Making the data-type configurable
        self._target_tensor = torch.nn.Parameter(
            torch.tensor(self._rank_targ, dtype=torch.float32).unsqueeze(1), requires_grad=False
        )
        
    def forward(
        self,
        outputs,
        labels,
        num_items_in_batch,
        scores,
        is_defeasible
    ) -> torch.Tensor:
        """ Different from normal token loss,
        we use a soft label [batch_size, seq_len, vocab_size] to calculate the loss.

        the first slice over the second_dim can be used to calculate a ignore_index mask.
        """
        
        logits = outputs['logits'] if isinstance(outputs, dict) else outputs[0]
        # mask = labels[:, :, 0].eq(self._ignore_index)  # [batch_size, 1, vocab_size]
        
        # calculate KL divergence between the soft label and the logits
        loss = self._compute_loss(
            logits=logits,
            is_defeasible=is_defeasible,
            labels=labels,
            scores=scores,
            num_items_in_batch=num_items_in_batch
        )
        
        score_loss = self._compute_score_loss(
            logits=logits,
            scores=scores,
            labels=labels,
            is_defeasible=is_defeasible,
            num_items_in_batch=num_items_in_batch
        )

        # print("-" * 20)
        # print(loss)
        # print(score_loss)
        # print("-" * 20)
        
        return loss + self._score_loss_scale * score_loss
        
    def _compute_score_loss(
        self,
        logits,
        scores,
        labels,
        is_defeasible,
        num_items_in_batch
    ):
        """ """
        
        batch_size = logits.size(0)
        # mask = ~labels.eq(self._ignore_index)
        mask = labels[:, :, 0].eq(self._ignore_index)
        # We use the trick that max will return the first index of the max value
        first_non_ignore_indices = torch.argmax((~mask).int(), dim=1)
        selected_logits = logits[torch.arange(batch_size), first_non_ignore_indices]  # [batch_size, vocab_size]

        pred_scores = torch.matmul(
            torch.nn.functional.softmax(selected_logits[..., self._selection_tensor], dim=-1),
            self._target_tensor
        ).view(-1, 2).contiguous()
        is_defeasible = is_defeasible.view(-1, 2)[:, 0].contiguous()
        pred_diff = pred_scores[:, 0] - pred_scores[:, 1]
        
        # print(pred_diff)
        
        scores = torch.matmul(
            scores,
            self._target_tensor
        ).view(-1, 2).contiguous()
        
        score_diff_sign = torch.sign(scores[:, 0] - scores[:, 1])
        pos_loss = torch.nn.functional.relu(self._margin - score_diff_sign * pred_diff).masked_fill(~is_defeasible, 0)
        
        return pos_loss.sum() / (is_defeasible.sum() + 1e-6)
        
    def _compute_loss(
        self,
        logits,
        is_defeasible,
        labels,
        scores,
        num_items_in_batch
    ) -> torch.Tensor:
        """ """

        # shift the labels to the right
        logits = logits[:, :-1, :].contiguous()
        labels = labels[:, 1:, :].contiguous()
        mask = labels[:, :, 0].eq(self._ignore_index)
        
        labels.masked_fill_(
            labels.eq(self._ignore_index),
            1.0
        )
        
        # pos_loss = torch.nn.functional.cross_entropy(
        #     logits.view(-1, logits.size(-1)),
        #     torch.clamp(labels, min=0).view(-1, labels.size(-1)),
        #     reduction='none'
        # )
        
        # logits: [batch_size, seq_len, vocab_size]
        # labels: [batch_size, seq_len, vocab_size]
        
        # assume that labels are the soft labels
        if not self._reverse_kl_loss:
            pos_loss = torch.nn.functional.kl_div(
                torch.log_softmax(logits / self._temperature, dim=-1),
                labels,
                reduction='none',
            )
        else:
            # reversed kl divergence
            pos_loss = torch.nn.functional.kl_div(
                torch.log(labels),
                torch.log_softmax(logits / self._temperature, dim=-1),
                log_target=True,
                reduction="none",
            )

        # pos_loss: [2 * batch_size, seq_len]
        pos_loss = pos_loss.sum(dim=-1)
        pos_loss.masked_fill_(mask, 0)
        # if is_defeasible, we calculate the defeasible loss instead
        pos_loss.masked_fill_(is_defeasible.unsqueeze(-1), 0)
        num_active_items = mask.numel() - mask.sum()
        loss = pos_loss.sum() / (num_active_items + 1e-6)
        
        return loss