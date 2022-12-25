from config import Config
from transformermodule.MLM import MLMTrainer

# Primary pipeline for training models on EHR sequences
if __name__ == '__main__':
    config_file = ["config.ini"]
    args = Config(file_path=config_file)

    if args.task == 'pre_train':
        trainer = MLMTrainer(args)
        trainer.train(args.max_epochs)
    elif args.task == 'fine_tune':
        # Do fine-tuning
        pass