local model_stem = std.split(std.extVar("MODEL_NAME"), "/")[1];
local num_labels = std.parseInt(std.extVar("NUM_LABELS"));
local reg_method = std.extVar("REGULARIZATION");
local scale = std.parseJson(std.extVar("SCALE_FACTOR"));
local possible_scale_tag = if reg_method == "null" then "" else "::scale=" + scale;

{
    type: "evaluation",
    input_dir: "task_outputs/training/sft-regression/" + model_stem + "::nl=" + num_labels + "::reg=" + reg_method + possible_scale_tag,
    num_labels: num_labels,
    dataset_map: [
        // dname: "task_outputs/dataset/" + dname + "-" + num_labels,
        [dname, "task_outputs/dataset/" + dname + (if std.startsWith(dname, "defeasible") then "" else "-" + num_labels)]
        for dname in ["unli", "gnli", "ecare", "entailmentbank", "defeasible-snli", "defeasible-atomic"]
    ],
    output_dir: "task_outputs/eval_results/sft-regression/" + model_stem + "::nl=" + num_labels + "::reg=" + reg_method + possible_scale_tag,
}