""" """

import click
import logging
from tasker import BaseTask
from src.tasks import *


logging.basicConfig(level=logging.DEBUG)


@click.command()
@click.option("--config-path", type=click.Path(exists=True), help="Path to the config file.")
def main(
    config_path
):
    """ """
    BaseTask.construct_and_run(config_path)
    
    
if __name__ == "__main__":
    main()