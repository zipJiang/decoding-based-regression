# Decoding-based Regression


## General Usage

The pipeline is composed of tasks. Each task performs a unique step with intermediate outputs that can be utilized by future tasks. If a task is not updated, it will not be re-run. Otherwise, running a task will re-run all its dependencies.

the script that runs a task is `scripts/run_task.py`. Notice that this can be used with `accelerate`

```bash
python scripts/run_task.py --config-path <path-to-config>
```

An example wrapper scripts to submit tasks that can be used with slurm can be fount at `runs/run_task.sh`. One may need to update a couple paths to make it work on brtx.

## Tasks

### 1. Data Preparation

At the moment the only supported dataset is `UNLI`. A sample config can be found at `configs/dataset/unli-dataset-preparation-complete.yaml`. You may implement your own preprocessor by extending `src.dataset_processors.base_dataset_processor.BaseDatasetProcessor`.

### 2. Vocab Extension

To simplify future processing, we extend the vocabulary and embeddings to include the dataset vocabulary as a separate step before actual SFT. A sample config can be found at `configs/preprocess/resize_embeddings.yaml`.

### 3. SFT

The SFT task can be launched with `accelerate`. An example usage can be found in `runs/run_task.sh`. An example config can be found at `configs/training/sft_regression.yaml`.

### 4. Evaluation

The evaluation task will evaluate the model on the test set. An example config can be found at `configs/eval/unli_sft_evaluation.yaml`. Notice that the `level_to_score_func` can potentially be updated to reflect latest token to score functions (and acutally need to if we use different levels -- See `TODO`).


<span style="color: red">Let me know if you need access to `tasker` repo.</span>