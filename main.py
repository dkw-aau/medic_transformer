from baselinemodule.Baseline import Baseline
from Utils.utils import set_seeds
from config import Config

# Primary pipeline for training models on EHR sequences
from transformermodule.Trainers.LOSTrainer import LOSTrainer
from transformermodule.Trainers.MLMTrainer import MLMTrainer

if __name__ == '__main__':
    config_file = ["config.ini"]
    args = Config(file_path=config_file)
    set_seeds(42)

    trainer = None
    if args.workload == 'los':
        trainer = LOSTrainer(args)
    elif args.workload == 'mlm':
        trainer = MLMTrainer(args)
    elif args.workload == 'base':
        trainer = Baseline(args)
    else:
        exit(f'Task {args.workload} not Implemented')

    trainer.train()
