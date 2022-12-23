from config import Config
from transformermodule.MLM import train_mlm

# Primary pipeline for training models on EHR sequences
if __name__ == '__main__':
    config_file = ["config.ini"]
    args = Config(file_path=config_file)

    if args.task == 'pre_train':
        train_mlm(args)
    elif args.task == 'fine_tune':
        # Do fine-tuning
        pass