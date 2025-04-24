# Always Tell Me the Odds

LLM-based Fine-grained Conditional Probability Estimation
[\[Huggingface Collection\]](https://huggingface.co/collections/Zhengping/always-tell-me-the-odds-6806b1e01cb76d8c7f3a33ef)

### Running Inference

`scripts/inference.py` provides a minimal example of how to run inference with the model. Make sure that you install the required dependencies first, and the current directory is in your `PYTHONPATH`.

```bash
pip install -r requirements.txt
export PYTHONPATH=$(pwd):$PYTHONPATH
python scripts/inference.py
```

### Other Tasks

Each of the step in training and evaluation is implemented in `src/tasks/`. To run a specific task, you can use the following command:

```bash

python scripts/run_task.py \
    --config-path <path_to_config>
```

All configurations are stored in `configs/`. Each configuration file is a `*.jsonnet` file, where some parameters need to be specified through
environment variables. For training/evaluation tasks (configs listed under `configs/training/` or `configs/evaluation/`), they are compatible with huggingface's `accelerate` library. You can use `accelerate config` to set up the environment variables, and then run the training/evaluation tasks with `accelerate launch scripts/run_task.py --config-path <path_to_config>`.


### Citation

```bibtex
```