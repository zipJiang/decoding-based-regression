import click
import json
import os


__DTST__ = [
    "maieutic-com2sense",
    "maieutic-creak",
    "maieutic-csqa2",
    "bird-com2sense",
    "bird-today",
]


@click.command()
@click.option('--input-path', '-i', type=click.Path(exists=True), help='Path to the input file')
def main(
    input_path,
):
    """ """
    
    all_results = [(
        "Model",
        "Reg",
        "Scale",
        "Num Labels",
        *__DTST__
    )]
    
    # generate a function that convert a float to .xxx three decimal places
    def t3(x):
        return "{:.3f}".format(x)
    
    def enquire_eval(dtst, subdir):
        if not os.path.exists(os.path.join(input_path, subdir, f"{dtst}.jsonl")):
            return '-'
        with open(os.path.join(input_path, subdir, f"{dtst}.jsonl"), 'r') as f:
            data = json.load(f)
            return t3(data['accuracy'])
    
    for subdir in os.listdir(input_path):
        # if subdir.startswith("Qwen2.5-14B"):
        fields = subdir.split("::")
        model_name = fields[0]
        num_labels = int(fields[1][3:])
        reg = fields[2][4:]
        reg = '-' if reg == 'null' else reg
        scale = '-' if len(fields) == 3 else int(fields[3][6:])
        
        #     data = json.load(f)
            # print('---')
            # print(subdir)
            # print("unli: ", data['unli']['evaluation']['spearman'])
            # print("gnli: ", data['gnli']['evaluation']['spearman'])
            # print("ecare: ", data['ecare']['evaluation']['spearman'])
            # print("entailmentbank: ", data['entailmentbank']['evaluation']['spearman'])
            # print('---')
            # all_results.append(
            #     (model_name, reg, scale, num_labels, t3(data['unli']['evaluation']['spearman']['correlation']), t3(data['gnli']['evaluation']['spearman']['correlation']), t3(data['entailmentbank']['evaluation']['spearman']['correlation']), t3(data["ecare"]['evaluation']['spearman']['correlation']))
            # )
        all_results.append((
            model_name,
            reg,
            scale,
            num_labels,
            *[enquire_eval(dtst, subdir) for dtst in __DTST__]
        ))
            
    for result in all_results:
        print('| ' + ' | '.join([str(r) for r in result]) + ' |')
    
    
if __name__ == '__main__':
    main()