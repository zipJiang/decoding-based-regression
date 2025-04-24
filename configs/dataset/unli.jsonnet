{
    type: "dataset-preparation",
    output_dir: "task_outputs/unli-dataset-preparation-completion",
    dataset_processor: {
        type: "unli",
        number_of_levels: std.parseInt(std.extVar("NUM_LABELS")),
        template: {
            type: "unli"
        }
    }
}