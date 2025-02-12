""" """

from transformers import AutoTokenizer


def get_tokenizer(
    name
):
    """Take a normal tokenizer and prepare it to
    a generally usable state.
    """
    
    tokenizer = AutoTokenizer.from_pretrained(name)

    if name == "meta-llama/Meta-Llama-3-8B-Instruct":
        tokenizer.pad_token = '<|end_of_text|>'

    return tokenizer