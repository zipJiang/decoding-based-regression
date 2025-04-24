"""An updated trainer that passes down raw
scoring for the decoder to use to calculate a
regression/ordinal-classification loss.
"""

from trl import SFTTrainer
from transformers.utils import is_peft_available
from transformers.trainer import _is_peft_model
from overrides import overrides


class DecoderBasedRegressionTrainer(SFTTrainer):
    """ """
    def __init__(
        self,
        compute_score_loss_func = None,
        compute_loss_func = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.compute_loss_func = compute_loss_func
        # self.compute_score_loss_func = compute_score_loss_func
        
        if compute_score_loss_func is not None:
            raise NotImplementedError("compute_score_loss_func is no longer supported.")
        
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
        
        # # peek into labels
        # labels = inputs.get("labels", None)
        scores = inputs.pop("scores", None)
        is_defeasible = inputs.pop("is_defeasible", None)
        additional_kwargs = {}
        
        if scores is not None:
            additional_kwargs["scores"] = scores
        if is_defeasible is not None:
            additional_kwargs["is_defeasible"] = is_defeasible

        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            # User-defined compute_loss function
            if self.compute_loss_func is not None:
                loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch, **additional_kwargs)
            else:
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if self.args.average_tokens_across_devices and self.model_accepts_loss_kwargs:
            loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss