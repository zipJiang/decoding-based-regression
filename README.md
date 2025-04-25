# Always Tell Me the Odds

LLM-based Fine-grained Conditional Probability Estimation
[\[Huggingface Collection\]](https://huggingface.co/collections/Zhengping/always-tell-me-the-odds-6806b1e01cb76d8c7f3a33ef)

## Installation

Ensure you install all required dependencies and add the current directory to your `PYTHONPATH`:

```bash
conda create -n conditional_prob_llm python=3.12
conda activate conditional_prob_llm
pip install -r requirements.txt
export PYTHONPATH=$(pwd):$PYTHONPATH
```

## Running Inference

Use `scripts/inference.py` for a minimal example of running inference with the model:

```bash
python scripts/inference.py
```

## Other Tasks

Task implementations for training and evaluation are available in `src/tasks/`. To run a specific task:

```bash
python scripts/run_task.py --config-path <path_to_config>
```

All configurations are stored in `configs/` as `*.jsonnet` files. Some parameters require specification through environment variables. 

For training and evaluation tasks (configs in `configs/training/` or `configs/evaluation/`), you can use Hugging Face's `accelerate` library:

1. Set up environment variables with `accelerate config`
2. Run tasks with `accelerate launch scripts/run_task.py --config-path <path_to_config>`

## Data Synthesis

To synthesize pseudo labels:
1. Use `ReasoningBasedProbExtractor` to generate LLM estimations via the vLLM backend
2. Apply agreement-based filtering using `/scripts/data_synthesis.py`

## Citation

```bibtex
```