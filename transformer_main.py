from transformermodule.LOSTrainer import LOSTrainer
from transformermodule.MLMTrainer import MLMTrainer
from Utils.utils import set_seeds
from config import Config

# Primary pipeline for training models on EHR sequences
if __name__ == '__main__':
    config_file = ["config.ini"]
    args = Config(file_path=config_file)
    set_seeds(42)

    if args.workload == 'los':
        trainer = LOSTrainer(args)
    elif args.workload == 'mlm':
        trainer = MLMTrainer(args)
    else:
        exit(f'Task {args.workload} not Implemented')

    trainer.train(args.max_epochs)