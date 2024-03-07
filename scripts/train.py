"""
This file runs the main training/val loop
"""
import os
import json
import sys

sys.path.append(".")
sys.path.append("..")

from src.options.train_options import TrainOptions
from src.training import Trainer
import warnings

warnings.filterwarnings("ignore")


def main():
    opts = TrainOptions().parse()
    os.makedirs(opts.exp_dir, exist_ok=True)

    opts_dict = vars(opts)
    with open(os.path.join(opts.exp_dir, 'opt.json'), 'w') as f:
        json.dump(opts_dict, f, indent=4, sort_keys=True)

    trainer = Trainer(opts)
    trainer.train()


if __name__ == '__main__':
    main()
