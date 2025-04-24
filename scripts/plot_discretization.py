""" """

import matplotlib.pyplot as plt
import numpy as np
import click
from src.utils.transforms import _discretize_gaussian


@click.command()
def main():
    """ """
    
    discretized = _discretize_gaussian(
        mean=np.array([0.02, 0.42, 0.49, 0.98], dtype=np.float32),
        std=np.array([0.05]),
        levels=np.array([[0.05 + i * 0.1 for i in range(10)]], dtype=np.float32)
    )  # shape [4, 10]
    
    fig, axs = plt.subplots(1, 4, figsize=(20, 4))
    x = np.linspace(-0.2, 1.2, 1000)
    means = [0.02, 0.42, 0.49, 0.98]
    std = 0.05
    levels = np.array([0.05 + i * 0.1 for i in range(10)])
    # Set integer y-ticks for all subplots
    for ax in axs:
        ax.yaxis.set_major_locator(plt.MaxNLocator(5, integer=True))
        
        ax.set_xticks(np.arange(0, 1.2, 0.2))
        ax.set_xlim(0, 1)

    for i, (ax, mu) in enumerate(zip(axs, means)):
        # Plot Gaussian PDF
        gaussian = np.exp(-0.5 * ((x - mu) / std) ** 2) / (std * np.sqrt(2 * np.pi))
        ax.plot(x, gaussian, 'b-', label='Gaussian')
        
        # Plot discretized distribution
        for j in range(len(levels)):
            # if j < len(levels) - 1:
            bar_height = discretized[i, j] / 0.1  # Normalize by bin width
            if j == 0:
                ax.bar(levels[j], bar_height, width=0.1, alpha=0.5, color='r', edgecolor='r', label='Discretized')
            else:
                ax.bar(levels[j], bar_height, width=0.1, alpha=0.5, color='r', edgecolor='r')
        
        ax.set_title(f'μ = {mu:.2f}, σ = {std:.2f}', fontsize=14)
        ax.legend(fontsize=14)
        ax.fill_between(x, gaussian, alpha=0.3, color='blue', label='Gaussian')
        ax.grid(True)

    plt.tight_layout()
    fig.savefig('discretization.pdf')
    
    
if __name__ == '__main__':
    main()