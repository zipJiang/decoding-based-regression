"""
"""

from typing import (
    Text,
    List,
    Tuple
)
from tqdm import tqdm
import click
import torch
from src.rank_dicts import SingleLabelRankDict
from src.chat_templates import (
    UNLITemplate
)
try:
    import ujson as json
except ImportError:
    import json
import logging
from transformers.pipelines import PIPELINE_REGISTRY
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, PreTrainedTokenizer
from peft import PeftModel, PeftConfig
from src.pipelines.level_to_score_pipeline import LevelToScorePipeline
import re
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


@click.command()
@click.option(
    '--input_dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
)
def main(
    input_dir,
):
    """ """
    
    def _parse_ckpt_dir(directory: Text) -> int:
        match = re.search(r"checkpoint-(\d+)", directory)
        return int(match.group(1))
    
    ckpts = []
    for ckpt_dir in os.listdir(input_dir):
        if not os.path.isdir(os.path.join(input_dir, ckpt_dir)):
            continue
        ckpt_num = (ckpt_dir, _parse_ckpt_dir(ckpt_dir))
        ckpts.append(ckpt_num)
        
    latest_ckpt_dir = sorted(ckpts, key=lambda x: x[1], reverse=True)[0][0]
    logger.info(f"Loading the state file checkpoint from {latest_ckpt_dir}.")
    
    with open(os.path.join(input_dir, latest_ckpt_dir, "trainer_state.json") ,'r', encoding='utf-8') as file_:
        training_state = json.load(file_)
        best_ckpt = training_state['best_model_checkpoint']
    logger.info(f"Best checkpoint is {best_ckpt}.")
    
    config = PeftConfig.from_pretrained(best_ckpt)
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,
        attn_implementation="flash_attention_2",
    )
    peft_model = PeftModel.from_pretrained(model, best_ckpt, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(best_ckpt)
    rank_dict = SingleLabelRankDict.from_tokenizer(tokenizer)
    num_labels = len(rank_dict)

    PIPELINE_REGISTRY.register_pipeline(
        "level-to-score",
        pipeline_class=LevelToScorePipeline,
        pt_model=AutoModelForCausalLM
    )
    
    def _level_to_score_func(
        logits: Tuple[torch.FloatTensor],
        tokenizer: PreTrainedTokenizer
    ) -> Tuple[List[float], List[float]]:
        """ """
        # TODO: factor the number_of_levels out as configurable
        # parameters
        logits = logits[0]
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
        model=peft_model,
        max_new_tokens=2,
        tokenizer=tokenizer,
        device=0,
        level_to_score_func=_level_to_score_func,
        torch_dtype=torch.bfloat16,
    )
    
    template = UNLITemplate()
    
    with open("data/inherit-disagreement.jsonl", 'r', encoding='utf-8') as file_:
        dataset = [json.loads(line) for line in file_]
        
    inputs = [
        template.get_prompt_template(
            **item
        ) + template.get_completion_template(is_completion=True)
        for item in dataset
    ]
    
    print(len(inputs))

    results = [pipe(ipt, do_sample=False)[0] for ipt in tqdm(inputs)]
    
    with open("data/analysis/inherit-disagreement.jsonl", 'w', encoding='utf-8') as file_:
        for item, result in zip(dataset, results):
            file_.write(
                json.dumps({
                    **item,
                    "result": result,
                }) + "\n"
            )
            
            
if __name__ == "__main__":
    main()