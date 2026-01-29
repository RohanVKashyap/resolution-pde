import logging
import wandb
from typing import Dict

"""Source code: https://github.com/RohanVKashyap/invariance-pde/blob/main/utils/log_utils.py"""

class WandbLogger(logging.Logger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_wandb = False
    
    def enable_wandb(self):
        self.use_wandb = True

    def wandb(self, log_dict : Dict, *args, **kwargs):
        if self.use_wandb:
            wandb.log(log_dict, *args, **kwargs)
    
    def wandb_summary(self, log_dict : Dict, *args, **kwargs):
        if self.use_wandb:
            wandb.run.summary.update(log_dict, *args, **kwargs)
        

logging.setLoggerClass(WandbLogger)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_logger(name=__name__, level=logging.INFO) -> WandbLogger:
    logging.setLoggerClass(WandbLogger)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger

def add_file_handler(logger, file_path, level=logging.INFO):
    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(level)
    logger.addHandler(file_handler)