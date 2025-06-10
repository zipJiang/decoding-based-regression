from vllm import LLM, SamplingParams

import torch
from transformers import (
    PreTrainedTokenizer
)
from typing import (
    Dict,
    List,
)
from src.rank_dicts import SingleLabelRankDict
from src.chat_templates import UNLITemplate

def _level_to_score_func_vllm(
    results,  # vLLM returns a list of dicts for each token position
    tokenizer: PreTrainedTokenizer,
    position: int = 0  # which token position to use for scoring
) -> Dict:
    """ 
    Adapted function for vLLM logprobs instead of HuggingFace logits
    Args:
        logprobs_list: List of dictionaries, each containing token_id -> Logprob mappings
        tokenizer: The tokenizer
        position: Which token position to use (0 = first generated token)
    """
    rank_dict = SingleLabelRankDict.from_tokenizer(tokenizer)
    num_labels = len(rank_dict)
    considering_ids = tokenizer.convert_tokens_to_ids([f" <|label_level_{i}|>" for i in range(num_labels)])

    res = []
    for result in results:
        logprobs_dict = result.outputs[0].logprobs[position]
        selective_logprobs = []
        for token_id in considering_ids:
            if token_id in logprobs_dict:
                # Access the logprob attribute from the Logprob object
                logprob_value = logprobs_dict[token_id].logprob
                selective_logprobs.append(logprob_value)
            else:
                selective_logprobs.append(float('-inf'))
        
        selective_probs = torch.exp(torch.tensor(selective_logprobs, dtype=torch.float32))
        selective_probs = selective_probs / selective_probs.sum()
        num_labels = len(selective_logprobs)
        step_size = 1 / num_labels
        expectation_values = torch.tensor([i * step_size + 0.5 * step_size for i in range(num_labels)])

        score = (selective_probs * expectation_values).sum().item()
        res.append({"scores": score, "selective_logprobs": selective_logprobs})
    return res


model = LLM(
    model="Zhengping/conditional-probability-regression",
    tensor_parallel_size=4,
)
tokenizer = model.get_tokenizer()
sampling_params = SamplingParams(logprobs=20)
template = UNLITemplate()

premise = "Sam is sleeping."
hypothesis = "Sam is awake."

inputs = template.get_prompt_template(premise=premise, hypothesis=hypothesis) +\
    template.get_completion_template(is_completion=True)

result = model.chat(inputs, sampling_params=sampling_params)
print(_level_to_score_func_vllm(result, tokenizer))

