from Utils.logger import Logger
from config import Config
from transformermodule.FineTuneTrainer import FineTuneTrainer
from transformermodule.HistoryTrainer import HistoryTrainer
from transformermodule.MLMTrainer import MLMTrainer
from Utils.utils import set_seeds

# Primary pipeline for training models on EHR sequences
if __name__ == '__main__':
    config_file = ["config.ini"]
    args = Config(file_path=config_file)
    set_seeds(42)

    args.logger = Logger(args)

    if args.task == 'history':
        trainer = HistoryTrainer(args)
    elif args.task == 'pre_train':
        trainer = MLMTrainer(args)
    elif args.task == 'fine_tune':
        trainer = FineTuneTrainer(args)
    else:
        exit(f'Task {args.task} not Implemented')

    trainer.train(args.max_epochs)