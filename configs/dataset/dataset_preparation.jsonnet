local number_of_levels = std.parseInt(std.extVar("NUM_LABELS"));

local all_configs = {
    "unli": {
        type: "unli",
        number_of_levels: number_of_levels,
        template: {
            type: "unli"
        }
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
};

local get_dataset_processor(task_name) = 
    assert std.objectHas(all_configs, task_name) : "Unknown task name: " + task_name;
    all_configs[task_name];

{
    type: "dataset-preparation",
    output_dir: "task_outputs/dataset/" + std.extVar("TASK_NAME") + "-" + number_of_levels,
    dataset_processor: get_dataset_processor(std.extVar("TASK_NAME")),
}