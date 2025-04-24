import transformers
import torch
from transformers.pipelines import PIPELINE_REGISTRY
from transformers import (
    pipeline,
    AutoModelForCausalLM,
    PreTrainedTokenizer
)
from typing import (
    Dict,
    Callable,
    Tuple,
    List,
)
from src.pipelines.level_to_score_pipeline import LevelToScorePipeline
from src.rank_dicts import SingleLabelRankDict
from src.chat_templates import UNLITemplate


model = transformers.AutoModelForCausalLM.from_pretrained(
    "Zhengping/conditional-probability-regression",
    torch_dtype="auto",
    attn_implementation="flash_attention_2",
)
tokenizer = transformers.AutoTokenizer.from_pretrained(
    "Zhengping/conditional-probability-regression",
)

rank_dict = SingleLabelRankDict.from_tokenizer(tokenizer)

PIPELINE_REGISTRY.register_pipeline(
    "level-to-score",
    pipeline_class=LevelToScorePipeline,
    pt_model=AutoModelForCausalLM
)

# This allows fine-grained labeling, the greedy decoding gives a coarse score,
# one can also attach their own level-to-score function to the pipeline, e.g. using UNLI
# label transformation to get it more binarized
def _level_to_score_func(
    logits: Tuple[torch.FloatTensor],
    tokenizer: PreTrainedTokenizer
) -> Tuple[List[float], List[float]]:
    """ """
    logits = logits[0]
    num_labels = len(rank_dict)
    considering_ids = tokenizer.convert_tokens_to_ids([f" <|label_level_{i}|>" for i in range(num_labels)])
    # print(considering_ids, logits.shape)
    selective_logits = torch.index_select(logits, 1, torch.tensor(considering_ids, device=logits.device))
    # print(selective_logits)
    step_size = 1 / num_labels
    expectation = torch.tensor([[i * step_size + 1 / 2 * step_size for i in range(num_labels)]], device=selective_logits.device)
    scores = torch.softmax(selective_logits, dim=-1) @ expectation.T
    scores = scores.squeeze(-1).tolist()
    # indices = torch.argmax(selective_logits, dim=-1).tolist()
    # scores = [i * 0.1 + 0.05 for i in indices]
    # print(scores)
    return scores, selective_logits.tolist()

pipe = pipeline(
    # "text-generation",
    "level-to-score",
    model=model,
    max_new_tokens=2,
    tokenizer=tokenizer,
    device=0,
    level_to_score_func=_level_to_score_func,
    torch_dtype=torch.bfloat16,
)

template = UNLITemplate()

premise = "Sam is sleeping."
hypothesis = "Sam is awake."

inputs = template.get_prompt_template(premise=premise, hypothesis=hypothesis) +\
    template.get_completion_template(is_completion=True)
    
result = pipe(inputs)
print(result)