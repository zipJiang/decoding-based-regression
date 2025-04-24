local model_stem = std.split(std.extVar("MODEL_NAME"), "/")[1];
local num_labels = std.parseInt(std.extVar("NUM_LABELS"));
local reg_method = std.extVar("REGULARIZATION");
local scale = std.parseJson(std.extVar("SCALE_FACTOR"));
local possible_scale_tag = if reg_method == "null" then "" else "::scale=" + scale;

// Regularization can be mse, margin and null

{
    type: "sft-regression",
    output_dir: "./task_outputs/training/sft-regression/" + model_stem + "::nl=" + num_labels + "::reg=" + reg_method + possible_scale_tag,
    input_dir: "/weka/scratch/bvandur1/zjiang31/decoding-based-regression/task_outputs/dataset/unli-" + num_labels,
    learning_rate: 0.00002,
    model_name: "/weka/scratch/bvandur1/zjiang31/decoding-based-regression/task_outputs/resized/" + model_stem + "-reb-" + num_labels,
    is_chat: true,
    [if reg_method != "null" && reg_method != "fd" then "rank_dict"]: {
        // This might need to be extended for more complex training setups
        "type": "single-label"
    },
    [if reg_method != "null" && reg_method != "fd" then "score_loss_func"]: {
        type: reg_method,
        scale_factor: scale
    },
    force_diffuse: reg_method == "fd",
}