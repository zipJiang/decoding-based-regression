local input_dir = std.extVar("INPUT_DIR");
local basename_list = std.split(input_dir, '/');
local basename = basename_list[std.length(basename_list) - 1];

{
    type: "structural-evaluation",
    // input_dir: "./task_outputs/training/sft-regression/" + model_stem + "::nl=" + num_labels + "::temp=" + temperature + "::reverse_kl=" + reverse_kl + "::std=" + standard_deviation + "::lsf=" + label_smoothing_factor + (if trust then "::trust" else ""),
    input_dir: input_dir,
    num_labels: 10,
    // output_dir: "task_outputs/struct_eval_results/" + (if use_encoder == 1 then "encoder" else "sft-regression/" + model_stem + "::nl=" + num_labels + "::reg=" + reg_method + possible_scale_tag),
    // output_dir: "./task_outputs/struct_eval_results/sft-regression/" + model_stem + "::nl=" + num_labels + "::temp=" + temperature + "::reverse_kl=" + reverse_kl + "::std=" + standard_deviation + "::lsf=" + label_smoothing_factor + (if trust then "::trust" else ""), 
    output_dir: "./task_outputs/struct_eval_results/_sft-regression/" + basename,
    tasks: [
        "maieutic-com2sense",
        "maieutic-creak",
        "maieutic-csqa2",
        "bird-com2sense",
        "bird-today",
    ]
}