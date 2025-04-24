local model_stem = std.split(std.extVar("MODEL_NAME"), "/")[1];
local temperature = std.extVar("TEMPERATURE");
local reverse_kl = std.parseInt(std.extVar("REVERSE_KL"));
local num_labels = std.parseInt(std.extVar("NUM_LABELS"));
local standard_deviation = std.extVar("STANDARD_DEVIATION");
local label_smoothing_factor = std.extVar("LABEL_SMOOTHING_FACTOR");
// local trust = std.parseInt(std.extVar("TRUST")) == 1;
// local trust = false;
local margin = std.parseJson(std.extVar("MARGIN"));
local scale_factor = std.parseJson(std.extVar("SCALE_FACTOR"));
local upsample = std.extVar("UPSAMPLE");
local downsample = std.extVar("DOWNSAMPLE");

{
    type: "evaluation",
    input_dir: "./task_outputs/training/dp-regression/" + model_stem + "::nl=" + num_labels + "::temp=" + std.extVar("TEMPERATURE") + "::reverse_kl=" + std.extVar("REVERSE_KL") + "::std=" + std.extVar("STANDARD_DEVIATION") + "::lsf=" + std.extVar("LABEL_SMOOTHING_FACTOR") + "::margin=" + std.extVar("MARGIN") + "::sc=" + std.extVar("SCALE_FACTOR") + "::up=" + upsample + "::down=" + downsample,
    num_labels: num_labels,
    dataset_map: [
        // dname: "task_outputs/dataset/" + dname + "-" + num_labels,
        [dname, "task_outputs/dataset/" + dname + (
            if std.startsWith(dname, "defeasible")
            || std.startsWith(dname, "circa")
            || std.startsWith(dname, "hellaswag")
            || std.startsWith(dname, "copa")
            then "" else "-" + num_labels
        )]
        for dname in ["unli", "gnli", "ecare", "entailmentbank", "defeasible-snli", "defeasible-atomic", "circa", "hellaswag", "copa"]
        // for dname in ["circa", "hellaswag", "copa"]
    ],
    output_dir: "./task_outputs/eval_results/dp-regression/" + model_stem + "::nl=" + num_labels + "::temp=" + std.extVar("TEMPERATURE") + "::reverse_kl=" + std.extVar("REVERSE_KL") + "::std=" + std.extVar("STANDARD_DEVIATION") + "::lsf=" + std.extVar("LABEL_SMOOTHING_FACTOR") + "::margin=" + std.extVar("MARGIN") + "::sc=" + std.extVar("SCALE_FACTOR") + "::up=" + upsample + "::down=" + downsample,
}