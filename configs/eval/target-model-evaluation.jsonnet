local input_dir = std.extVar("INPUT_DIR");
local basename_list = std.split(input_dir, '/');
local basename = basename_list[std.length(basename_list) - 1];


{
    type: "evaluation",
    // input_dir: "./task_outputs/training/sft-regression/" + model_stem + "::nl=" + num_labels + "::temp=" + temperature + "::reverse_kl=" + reverse_kl + "::std=" + standard_deviation + "::lsf=" + label_smoothing_factor + (if trust then "::trust" else ""),
    input_dir: input_dir,
    num_labels: 10,
    dataset_map: [
        // dname: "task_outputs/dataset/" + dname + "-" + num_labels,
        [dname, "task_outputs/dataset/" + dname + (
            if std.startsWith(dname, "defeasible")
            || std.startsWith(dname, "circa")
            || std.startsWith(dname, "hellaswag")
            || std.startsWith(dname, "copa")
            then "" else "-" + 10
        )]
        for dname in ["unli", "gnli", "ecare", "entailmentbank", "defeasible-snli", "defeasible-atomic", "circa", "hellaswag", "copa"]
        // for dname in ["circa", "hellaswag", "copa"]
    ],
    // output_dir: "./task_outputs/eval_results/sft-regression/" + model_stem + "::nl=" + num_labels + "::temp=" + temperature + "::reverse_kl=" + reverse_kl + "::std=" + standard_deviation + "::lsf=" + label_smoothing_factor + (if trust then "::trust" else ""),
    output_dir: "./task_outputs/eval_results/_sft-regression/" + basename,
}