"""
"""
import click
from tqdm import tqdm
import numpy as np
try:
    import ujson as json
except ImportError:
    import json

import matplotlib.pyplot as plt
    

def main():
    """"""
    with open("data/analysis/inherit-disagreement.jsonl", 'r', encoding='utf-8') as file_:
        dataset = [json.loads(line) for line in file_]
        
    for idx, item in enumerate(tqdm(dataset)):
        
        if not idx in [14, 22, 24, 27, 40, 62]:
            continue
        
        labels = item['labels']
        labels = np.clip(
            np.array(labels),
            a_min=-50,
            a_max=50,
        )
        
        premise = item['premise']
        hypothesis = item['hypothesis']

        # normalize to [0, 10]
        labels = (labels + 50) / 10
        # get frequency of labels in each bin
        bin_counts = np.zeros(10)
        for label in labels:
            bin_idx = int(label)
            if bin_idx == 10:  # handle edge case for maximum value
                bin_idx = 9
            bin_counts[bin_idx] += 1
        frequencies = bin_counts / len(labels)
        
        
        # create figure for new plot
        fig, ax = plt.subplots()
        # ax.text(0.05, -0.5, f'P: {premise}\nh: {hypothesis}', 
        #     transform=ax.transAxes, fontsize=14, 
        #     verticalalignment='top', wrap=True)
        
        
        # bin labels to 10 bins

        # bar plot for frequencies
        bin_edges = np.arange(11)  # 0 to 10
        ax.bar(bin_edges[:-1] + 0.5, frequencies, width=1.0, align='center', alpha=0.5, color='r', edgecolor='r', label='Human')

        ax.set_xticks([0.5 + i for i in range(10)])
        ax.set_xticklabels([f"Bin {i}" for i in range(10)])
        ax.set_xlabel('Labels', fontsize=14)
        ax.set_ylabel('Density', fontsize=14)
        # save figure
        
        logits = np.array(item['result']['selective_logits'])
        # softmax normalization
        logits = logits - np.max(logits, axis=-1, keepdims=True)
        probs = np.exp(logits)
        probs = probs / np.sum(probs, axis=-1, keepdims=True)
        
        # plot probabilities as a line on the same axis
        ax.plot(bin_edges[:-1] + (bin_edges[1] - bin_edges[0])/2, probs, color='blue', marker='o', linestyle='-', label='Model')
        ax.legend(fontsize=14)
        plt.tight_layout()
        plt.savefig(f'data/analysis/plots/hist_{idx}.pdf')
        plt.close()
        


if __name__ == '__main__':
    main()