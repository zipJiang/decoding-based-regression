local model_stem = std.split(std.extVar("MODEL_NAME"), "/")[1];
local use_encoder = std.parseInt(std.extVar("USE_ENCODER"));
local num_labels = std.parseInt(std.extVar("NUM_LABELS"));
local reg_method = std.extVar("REGULARIZATION");
local scale = std.parseJson(std.extVar("SCALE_FACTOR"));
local possible_scale_tag = if reg_method == "null" then "" else "::scale=" + scale;

{
    type: "structural-evaluation",
    input_dir: if use_encoder == 1 then null else "task_outputs/training/sft-regression/" + model_stem + "::nl=" + num_labels + "::reg=" + reg_method + possible_scale_tag,
    num_labels: num_labels,
    output_dir: "task_outputs/struct_eval_results/" + (if use_encoder == 1 then "encoder" else "sft-regression/" + model_stem + "::nl=" + num_labels + "::reg=" + reg_method + possible_scale_tag),
    tasks: [
        "maieutic-com2sense",
        "maieutic-creak",
        "maieutic-csqa2",
        "bird-com2sense",
        "bird-today",
    ]
}