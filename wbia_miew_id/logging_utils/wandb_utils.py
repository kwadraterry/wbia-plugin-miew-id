import wandb
import os
from dotenv import load_dotenv



def init_wandb(exp_name, project_name, config=None):

    print('Initializing wandb run')
    print('exp_name:', exp_name)
    print('project_name:', project_name)

    export_config = dict(config) if config else None

    run = wandb.init(
        project=project_name, 
        name=exp_name, 
        config=export_config,
        reinit=True
        )
    # wandb.config = config  # {"learning_rate": 0.001, "epochs": 100, "batch_size": 128}

    return run


def finish_wandb():
    wandb.finish()


class WandbContext:
    def __init__(self, config):
        self.config = config

    def __enter__(self):
        if self.config.engine.use_wandb:
            load_dotenv()
            init_wandb(self.config.exp_name, self.config.project_name, config=self.config)

    def __exit__(self, type, value, traceback):
        if self.config.engine.use_wandb:
            finish_wandb()
