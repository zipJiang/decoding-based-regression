local number_of_levels = std.parseInt(std.extVar("NUM_LABELS"));
local with_other_tests = std.parseInt(std.extVar("WITH_OTHER_TESTS")) == 1;

local _convert(x) = if x == "defeasible-pseudo-label" then "defeasible-pseudo-label" + "-" + std.extVar("UPSAMPLE") + "-" + std.extVar("DOWNSAMPLE") else x;

local all_configs = {
    "unli": {
        type: "unli",
        number_of_levels: number_of_levels,
        template: {
            type: "unli"
        }
    },
    "defeasible-snli": {
        type: "defeasible-nli",
        template: {
            type: "unli"
        },
        subset: "snli"
    },
    "defeasible-atomic": {
        type: "defeasible-nli",
        template: {
            type: "unli"
        },
        subset: "atomic"
    },
    "ecare": {
        type: "multi-premise",
        datapath: "data/synthetic/Ecare_test.jsonl",
        number_of_levels: number_of_levels,
        template: {
            type: "unli"
        }
    },
    "entailmentbank": self["ecare"] + {
        datapath: "data/synthetic/EntailmentBank_test.jsonl"
    },
    "gnli": self["ecare"] + {
        datapath: "data/synthetic/GNLI_test.jsonl"
    },
    "pseudo-label": {
        type: "pseudo-label",
        // number_of_levels: number_of_levels,
        template: {
            type: "unli"
        },
        data_dir: "data/pseudo-labeled",
        model_names: [
            "DeepSeek/DeepSeek-R1-Distill-Qwen-32B",
            "meta-llama/Llama-3.3-70B-Instruct",
            "Qwen/Qwen2.5-72B-Instruct-AWQ",
            "Qwen/QwQ-32B",
        ],
        dataset_names: [
            "anli",
            "wanli"
        ],
        include_other_tests: with_other_tests,
        unli_upsample: std.parseInt(std.extVar("UPSAMPLE")),
    },
    "defeasible-training": {
        type: "defeasible-training",
        template: {
            type: "unli"
        },
        delta_nli_subsets: [
            "atomic",
            "snli"
        ],
        seed: 2265,
        mixed_dev: true,
    },
    "defeasible-pseudo-label": {
        type: "defeasible-pseudo-label",
        template: {
            type: "unli"
        },
        data_dir: "data/confidence_aggregate",
        delta_nli_subsets: [
            "atomic",
            "snli"
        ],
        dataset_names: [
            "anli",
            "wanli"
        ],
        up_sample: std.parseInt(std.extVar("UPSAMPLE")),
        down_sample: std.parseInt(std.extVar("DOWNSAMPLE")),
        seed: 42,
        mixed_dev: with_other_tests
    },
    "pseudo-label-trust": {
        type: "pseudo-label",
        // number_of_levels: number_of_levels,
        template: {
            type: "unli"
        },
        data_dir: "data/pseudo-labeled",
        model_names: [
            "DeepSeek/DeepSeek-R1-Distill-Qwen-32B",
            "meta-llama/Llama-3.3-70B-Instruct",
            "Qwen/Qwen2.5-72B-Instruct-AWQ",
            "Qwen/QwQ-32B",
        ],
        dataset_names: [
            "anli",
            "wanli"
        ],
        include_other_tests: with_other_tests,
        unli_upsample: std.parseInt(std.extVar("UPSAMPLE")),
        trust_nli_label: true
    },
    "copa": {
        type: "copa",
        template: {
            type: "unli"
        },
    },
    "hellaswag": {
        type: "hellaswag",
        template: {
            type: "unli"
        },
    },
    "circa": {
        type: "circa",
        datapath: "data/circa.jsonl",
        template: {
            type: "unli"
        },
    },
};

local get_dataset_processor(task_name) = 
    assert std.objectHas(all_configs, task_name) : "Unknown task name: " + task_name;
    all_configs[task_name];

{
    type: "dataset-preparation",
    output_dir: "task_outputs/dataset/" + _convert(std.extVar("TASK_NAME")) + (
        if std.startsWith(std.extVar("TASK_NAME"), "defeasible")
        || std.startsWith(std.extVar("TASK_NAME"), "pseudo")
        || std.startsWith(std.extVar("TASK_NAME"), "copa")
        || std.startsWith(std.extVar("TASK_NAME"), "hellaswag")
        || std.startsWith(std.extVar("TASK_NAME"), "circa")
        then "" else "-" + number_of_levels
    ),
    dataset_processor: get_dataset_processor(std.extVar("TASK_NAME")),
}