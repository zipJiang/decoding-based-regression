"""Different from single_token_reg_loss,
the logits isn't selected.
"""

from abc import ABC, abstractmethod
from overrides import overrides
import torch
from registrable import Registrable
from dataclasses import dataclass
from typing import List, Dict


class SoftTokenLoss(torch.nn.Module):
    """ """

    _ignore_index: int = -100
    
    def __init__(self):
        """ """
        super().__init__()
        
    def forward(
        self,
        outputs,
        labels,
        num_items_in_batch
    ) -> torch.Tensor:
        """ Different from normal token loss,
        we use a soft label [batch_size, seq_len, vocab_size] to calculate the loss.

        the first slice over the second_dim can be used to calculate a ignore_index mask.
        """
        
        logits = outputs['logits'] if isinstance(outputs, dict) else outputs[0]
        mask = labels[:, :, 0].eq(self._ignore_index)  # [batch_size, 1, vocab_size]
        
        # calculate KL divergence between the soft label and the logits
        return self._compute_loss(logits, labels, mask, num_items_in_batch)
        
    def _compute_loss(
        self,
        logits,
        labels,
        mask,
        num_items_in_batch
    ) -> torch.Tensor:
        """ """
        
        pos_loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1, labels.size(-1)),
            reduction='none'
        )

        # We only calculate the loss for the non-masked tokens
        pos_loss.masked_fill_(mask.flatten(), 0)
        num_active_items = mask.numel() - mask.sum()

        loss = pos_loss.sum() / num_active_items
        
        return loss