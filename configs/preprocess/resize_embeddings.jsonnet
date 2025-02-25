local model_name = std.extVar("MODEL_NAME");
local size = std.parseInt(std.extVar("NUM_LABELS"));

{
    type: "resize-embeddings",
    model_name: model_name,
    number_of_levels: size,
    output_dir: "task_outputs/resized/" + std.split(model_name, "/")[1] + "-reb-" + size,
}