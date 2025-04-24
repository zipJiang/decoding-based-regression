"""
"""

import os
import click
import datasets


@click.command()
def main():
    """ """

    dataset = datasets.load_from_disk(
        os.path.join(
            "task_outputs/dataset/pseudo-label/dataset"
        )
    )
    
    # push to hub
    dataset.push_to_hub(
        repo_id="Zhengping/UNLI-style-synthetic",
        token="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"
    )
    
    
if __name__ == "__main__":
    main()