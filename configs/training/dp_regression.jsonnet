local model_stem = std.split(std.extVar("MODEL_NAME"), "/")[1];
local temperature = std.parseJson(std.extVar("TEMPERATURE"));
local reverse_kl = std.parseInt(std.extVar("REVERSE_KL")) == 1;
local num_labels = std.parseInt(std.extVar("NUM_LABELS"));
local standard_deviation = std.parseJson(std.extVar("STANDARD_DEVIATION"));
local label_smoothing_factor = std.parseJson(std.extVar("LABEL_SMOOTHING_FACTOR"));
// local trust = std.parseInt(std.extVar("TRUST")) == 1;
local margin = std.parseJson(std.extVar("MARGIN"));
local scale_factor = std.parseJson(std.extVar("SCALE_FACTOR"));
local upsample = std.extVar("UPSAMPLE");
local downsample = std.extVar("DOWNSAMPLE");
// local reg_method = std.extVar("REGULARIZATION");
// local scale = std.parseJson(std.extVar("SCALE_FACTOR"));
// local possible_scale_tag = if reg_method == "null" then "" else "::scale=" + scale;

// Regularization can be mse, margin and null

{
    type: "defeasible-regression",
    margin: margin,
    output_dir: "./task_outputs/training/dp-regression/" + model_stem + "::nl=" + num_labels + "::temp=" + std.extVar("TEMPERATURE") + "::reverse_kl=" + std.extVar("REVERSE_KL") + "::std=" + std.extVar("STANDARD_DEVIATION") + "::lsf=" + std.extVar("LABEL_SMOOTHING_FACTOR") + "::margin=" + std.extVar("MARGIN") + "::sc=" + std.extVar("SCALE_FACTOR") + "::up=" + upsample + "::down=" + downsample,
    input_dir: "/weka/scratch/bvandur1/zjiang31/decoding-based-regression/task_outputs/dataset/defeasible-pseudo-label-" + upsample + "-" + downsample,
    learning_rate: 0.00002,
    label_smoothing_factor: label_smoothing_factor,
    model_name: "/weka/scratch/bvandur1/zjiang31/decoding-based-regression/task_outputs/resized/" + model_stem + "-reb-" + num_labels,
    loss_temperature: temperature,
    reverse_kl_loss: reverse_kl,
    std: standard_deviation,
    scale_factor: scale_factor,
}