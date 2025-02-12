"""
"""

import enum
import torch
from typing import Dict, Callable, Tuple, List
from transformers import Pipeline, TextGenerationPipeline
from transformers import PreTrainedTokenizer
from transformers.pipelines import PIPELINE_REGISTRY
from transformers.pipelines.text_generation import Chat, ReturnType


class LevelToScorePipeline(TextGenerationPipeline):
    
    def __init__(
        self,
        level_to_score_func: Callable[[Tuple[torch.FloatTensor], PreTrainedTokenizer], List[float]],
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._level_to_score_func = level_to_score_func
        
    def preprocess(
        self,
        prompt_text,
        prefix="",
        handle_long_generation=None,
        add_special_tokens=None,
        truncation=None,
        padding=None,
        max_length=None,
        continue_final_message=None,
        **generate_kwargs,
    ):
        # Only set non-None tokenizer kwargs, so as to rely on the tokenizer's defaults
        tokenizer_kwargs = {
            "add_special_tokens": add_special_tokens,
            "truncation": truncation,
            "padding": padding,
            "max_length": max_length,
        }
        tokenizer_kwargs = {key: value for key, value in tokenizer_kwargs.items() if value is not None}

        if isinstance(prompt_text, Chat):
            tokenizer_kwargs.pop("add_special_tokens", None)  # ignore add_special_tokens on chats
            # If the user passes a chat that ends in an assistant message, we treat it as a prefill by default
            # because very few models support multiple separate, consecutive assistant messages
            if continue_final_message is None:
                continue_final_message = prompt_text.messages[-1]["role"] == "assistant"
            inputs = self.tokenizer.apply_chat_template(
                prompt_text.messages,
                add_generation_prompt=not continue_final_message,
                continue_final_message=continue_final_message,
                return_dict=True,
                return_tensors=self.framework,
                **tokenizer_kwargs,
            )
        else:
            inputs = self.tokenizer(prefix + prompt_text, return_tensors=self.framework, **tokenizer_kwargs)

        inputs["prompt_text"] = prompt_text

        if handle_long_generation == "hole":
            cur_len = inputs["input_ids"].shape[-1]
            if "max_new_tokens" in generate_kwargs:
                new_tokens = generate_kwargs["max_new_tokens"]
            else:
                new_tokens = generate_kwargs.get("max_length", self.generation_config.max_length) - cur_len
                if new_tokens < 0:
                    raise ValueError("We cannot infer how many new tokens are expected")
            if cur_len + new_tokens > self.tokenizer.model_max_length:
                keep_length = self.tokenizer.model_max_length - new_tokens
                if keep_length <= 0:
                    raise ValueError(
                        "We cannot use `hole` to handle this generation the number of desired tokens exceeds the"
                        " models max length"
                    )

                inputs["input_ids"] = inputs["input_ids"][:, -keep_length:]
                if "attention_mask" in inputs:
                    inputs["attention_mask"] = inputs["attention_mask"][:, -keep_length:]

        return inputs
    
    def _forward(self, model_inputs, **generate_kwargs):
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs.get("attention_mask", None)
        # Allow empty prompts
        if input_ids.shape[1] == 0:
            input_ids = None
            attention_mask = None
            in_b = 1
        else:
            in_b = input_ids.shape[0]
        prompt_text = model_inputs.pop("prompt_text")

        # If there is a prefix, we may need to adjust the generation length. Do so without permanently modifying
        # generate_kwargs, as some of the parameterization may come from the initialization of the pipeline.
        prefix_length = generate_kwargs.pop("prefix_length", 0)
        if prefix_length > 0:
            has_max_new_tokens = "max_new_tokens" in generate_kwargs or (
                "generation_config" in generate_kwargs
                and generate_kwargs["generation_config"].max_new_tokens is not None
            )
            if not has_max_new_tokens:
                generate_kwargs["max_length"] = generate_kwargs.get("max_length") or self.generation_config.max_length
                generate_kwargs["max_length"] += prefix_length
            has_min_new_tokens = "min_new_tokens" in generate_kwargs or (
                "generation_config" in generate_kwargs
                and generate_kwargs["generation_config"].min_new_tokens is not None
            )
            if not has_min_new_tokens and "min_length" in generate_kwargs:
                generate_kwargs["min_length"] += prefix_length

        # User-defined `generation_config` passed to the pipeline call take precedence
        if "generation_config" not in generate_kwargs:
            generate_kwargs["generation_config"] = self.generation_config
            
        generate_kwargs["output_scores"] = not generate_kwargs.get("do_sample", False)
        generate_kwargs["return_dict_in_generate"] = True

        generated_sequence = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, **generate_kwargs)
        
        logits = None
        
        # TODO: check good default
        if generate_kwargs.get("return_scores", True):
            assert not generate_kwargs.get("do_sample", False), "return_logits=True is only supported for do_sample=False"
            
            # Proceed to process logits and convert to score average.
            # next_token_logits is [batch_size, vocab_size]
            # raw_logits is a tuple of ([next_token_logits, past_key_values])

            logits = generated_sequence.scores

        out_b = generated_sequence.sequences.shape[0]
        if self.framework == "pt":
            generated_sequence = generated_sequence.sequences.reshape(in_b, out_b // in_b, *generated_sequence.sequences.shape[1:])
        # elif self.framework == "tf":
        #     generated_sequence = tf.reshape(generated_sequence, (in_b, out_b // in_b, *generated_sequence.shape[1:]))
        return {"generated_sequence": generated_sequence, "input_ids": input_ids, "prompt_text": prompt_text, "logits": logits}
    
    def postprocess(
        self,
        model_outputs,
        return_type=ReturnType.FULL_TEXT,
        clean_up_tokenization_spaces=True,
        continue_final_message=None,
    ):
        generated_sequence = model_outputs["generated_sequence"][0]
        input_ids = model_outputs["input_ids"]
        prompt_text = model_outputs["prompt_text"]
        logits = model_outputs["logits"]
        scores = self._level_to_score_func(logits, self.tokenizer)
        
        generated_sequence = generated_sequence.numpy().tolist()
        records = []
        for sequence in generated_sequence:
            if return_type == ReturnType.TENSORS:
                record = {"generated_token_ids": sequence}
            elif return_type in {ReturnType.NEW_TEXT, ReturnType.FULL_TEXT}:
                # Decode text
                text = self.tokenizer.decode(
                    sequence,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                )

                # Remove PADDING prompt of the sequence if XLNet or Transfo-XL model is used
                if input_ids is None:
                    prompt_length = 0
                else:
                    prompt_length = len(
                        self.tokenizer.decode(
                            input_ids[0],
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                        )
                    )

                all_text = text[prompt_length:]
                if return_type == ReturnType.FULL_TEXT:
                    if isinstance(prompt_text, str):
                        all_text = prompt_text + all_text
                    elif isinstance(prompt_text, Chat):
                        if continue_final_message is None:
                            # If the user passes a chat ending in an assistant message, we treat it as a prefill by
                            # default because very few models support multiple separate, consecutive assistant messages
                            continue_final_message = prompt_text.messages[-1]["role"] == "assistant"
                        if continue_final_message:
                            # With assistant prefill, concat onto the end of the last message
                            all_text = list(prompt_text.messages)[:-1] + [
                                {
                                    "role": prompt_text.messages[-1]["role"],
                                    "content": prompt_text.messages[-1]["content"] + all_text,
                                }
                            ]
                        else:
                            # When we're not starting from a prefill, the output is a new assistant message
                            all_text = list(prompt_text.messages) + [{"role": "assistant", "content": all_text}]
                record = {"generated_text": all_text, "score": scores[0]}
            records.append(record)

        return records