// local model_stem = std.split(std.extVar("MODEL_NAME"), "/")[1];
local model_stem = "Qwen2.5-7B-Instruct";
local num_labels = 10;
local reg_method = "null";
local scale = 1;
local possible_scale_tag = if reg_method == "null" then "" else "::scale=" + scale;

// Regularization can be mse, margin and null

{
    type: "sft-regression",
    output_dir: "./task_outputs/training/sft-regression/" + model_stem + "::nl=" + num_labels + "::reg=" + reg_method + possible_scale_tag,
    input_dir: "/weka/scratch/bvandur1/zjiang31/decoding-based-regression/task_outputs/dataset/unli-" + num_labels,
    learning_rate: 0.00002,
    model_name: "/weka/scratch/bvandur1/zjiang31/decoding-based-regression/task_outputs/resized/" + model_stem + "-reb-" + num_labels,
    is_chat: true,
    force_diffuse: false,
}