local model_stem = std.split(std.extVar("MODEL_NAME"), "/")[1];
local temperature = std.extVar("TEMPERATURE");
local reverse_kl = std.parseInt(std.extVar("REVERSE_KL"));
local num_labels = std.parseInt(std.extVar("NUM_LABELS"));
local standard_deviation = std.extVar("STANDARD_DEVIATION");
local label_smoothing_factor = std.extVar("LABEL_SMOOTHING_FACTOR");
local upsample = std.extVar("UPSAMPLE");
local downsample = std.extVar("DOWNSAMPLE");

{
    type: "structural-evaluation",
    input_dir: "./task_outputs/training/dp-regression/" + model_stem + "::nl=" + num_labels + "::temp=" + std.extVar("TEMPERATURE") + "::reverse_kl=" + std.extVar("REVERSE_KL") + "::std=" + std.extVar("STANDARD_DEVIATION") + "::lsf=" + std.extVar("LABEL_SMOOTHING_FACTOR") + "::margin=" + std.extVar("MARGIN") + "::sc=" + std.extVar("SCALE_FACTOR") + "::up=" + upsample + "::down=" + downsample,
    num_labels: num_labels,
    // output_dir: "task_outputs/struct_eval_results/" + (if use_encoder == 1 then "encoder" else "sft-regression/" + model_stem + "::nl=" + num_labels + "::reg=" + reg_method + possible_scale_tag),
    output_dir: "./task_outputs/struct_eval_results/dp-regression/" + model_stem + "::nl=" + num_labels + "::temp=" + std.extVar("TEMPERATURE") + "::reverse_kl=" + std.extVar("REVERSE_KL") + "::std=" + std.extVar("STANDARD_DEVIATION") + "::lsf=" + std.extVar("LABEL_SMOOTHING_FACTOR") + "::margin=" + std.extVar("MARGIN") + "::sc=" + std.extVar("SCALE_FACTOR") + "::up=" + upsample + "::down=" + downsample,
    tasks: [
        "maieutic-com2sense",
        "maieutic-creak",
        "maieutic-csqa2",
        "bird-com2sense",
        "bird-today",
    ]
}