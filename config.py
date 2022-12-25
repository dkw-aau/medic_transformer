import configparser
import os


class Config:
    def __init__(self, file_path, path=os.getcwd()):
        self.seed = 42

        conf = configparser.ConfigParser()

        try:
            conf.read(file_path)
        except:
            exit("Could not read dataset module config.ini file")

        self.task = conf.get('general', 'task')

        # Data paths
        self.path = {
            'data_fold': path + '/Data/',
            'out_fold': path + '/Outputs/',
        }

        # File names
        self.corpus_name = conf.get('files', 'corpus_name')
        self.pretrain_name = conf.get('files', 'pretrain_name')
        self.finetune_name = conf.get('files', 'finetune_name')

        if self.task == 'corpus':
            self.prepare_parquet = conf.getboolean('extraction', 'prepare_parquet')
            self.max_sequences = conf.getint('extraction', 'max_sequences')

        elif self.task in ['pre_train', 'fine_tune']:

            self.step_eval = conf.getint('globals', 'step_eval')

            self.lr = conf.getfloat('optimization', 'lr')
            self.warmup_proportion = conf.getfloat('optimization', 'warmup_proportion')
            self.weight_decay = conf.getfloat('optimization', 'weight_decay')

            self.max_epochs = conf.getint('train_params', 'max_epochs')
            self.batch_size = conf.getint('train_params', 'batch_size')
            self.max_len_seq = conf.getint('train_params', 'max_len_seq')
            use_gpu = conf.getboolean('train_params', 'use_gpu')

            self.device = 'gpu' if use_gpu is True else 'cpu'

            self.hidden_size = conf.getint('model_params', 'hidden_size')
            self.layer_dropout = conf.getfloat('model_params', 'layer_dropout')
            self.num_hidden_layers = conf.getint('model_params', 'num_hidden_layers')
            self.num_attention_heads = conf.getint('model_params', 'num_attention_heads')
            self.att_dropout = conf.getfloat('model_params', 'att_dropout')
            self.intermediate_size = conf.getint('model_params', 'intermediate_size')
            self.hidden_act = conf.get('model_params', 'hidden_act')
            self.initializer_range = conf.getfloat('model_params', 'initializer_range')

    def __repr__(self):
        return f'Task: {self.task}'
