local model_stem = std.split(std.extVar("MODEL_NAME"), "/")[1];
local temperature = std.extVar("TEMPERATURE");
local reverse_kl = std.parseInt(std.extVar("REVERSE_KL"));
local num_labels = std.parseInt(std.extVar("NUM_LABELS"));
local standard_deviation = std.extVar("STANDARD_DEVIATION");
local label_smoothing_factor = std.extVar("LABEL_SMOOTHING_FACTOR");
local trust = std.parseInt(std.extVar("TRUST")) == 1;
// local trust = false;
local margin = std.parseJson(std.extVar("MARGIN"));
local scale_factor = std.parseJson(std.extVar("SCALE_FACTOR"));

{
    type: "structural-evaluation",
    input_dir: "./task_outputs/training/defeasible-regression/" + model_stem + "::nl=" + num_labels + "::temp=" + std.extVar("TEMPERATURE") + "::reverse_kl=" + std.extVar("REVERSE_KL") + "::std=" + std.extVar("STANDARD_DEVIATION") + "::lsf=" + std.extVar("LABEL_SMOOTHING_FACTOR") + "::margin=" + std.extVar("MARGIN") + "::sc=" + std.extVar("SCALE_FACTOR") + (if trust then "::trust" else ""),
    num_labels: num_labels,
    // output_dir: "task_outputs/struct_eval_results/" + (if use_encoder == 1 then "encoder" else "sft-regression/" + model_stem + "::nl=" + num_labels + "::reg=" + reg_method + possible_scale_tag),
    output_dir: "./task_outputs/struct_eval_results/defeasible-regression/" + model_stem + "::nl=" + num_labels + "::temp=" + std.extVar("TEMPERATURE") + "::reverse_kl=" + std.extVar("REVERSE_KL") + "::std=" + std.extVar("STANDARD_DEVIATION") + "::lsf=" + std.extVar("LABEL_SMOOTHING_FACTOR") + "::margin=" + std.extVar("MARGIN") + "::sc=" + std.extVar("SCALE_FACTOR") + (if trust then "::trust" else ""),
    tasks: [
        "maieutic-com2sense",
        "maieutic-creak",
        "maieutic-csqa2",
        "bird-com2sense",
        "bird-today",
    ]
}