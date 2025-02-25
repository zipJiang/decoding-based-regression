"""An updated trainer that passes down raw
scoring for the decoder to use to calculate a
regression/ordinal-classification loss.
"""

from trl import SFTTrainer
from overrides import overrides


class DecoderBasedRegressionTrainer(SFTTrainer):
    """ """
    def __init__(
        self,
        compute_score_loss_func,
        compute_loss_func = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.compute_loss_func = compute_loss_func
        self.compute_score_loss_func = compute_score_loss_func
        
    @overrides
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None
    ):
        """
        Compute training loss and additionally compute token accuracies
        """
        
        # peek into labels
        labels = inputs.get("labels", None)
        
        (loss, outputs) = super().compute_loss(
            model,
            inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch
        )

        scores = inputs.pop("scores", None)
        
        if scores is not None and self.compute_score_loss_func is not None:
            score_loss = self.compute_score_loss_func(outputs, labels, scores, num_items_in_batch)
            loss += score_loss
            
        return (loss, outputs) if return_outputs else loss