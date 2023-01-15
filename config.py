import configparser
from decouple import config as get_env
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
        self.history_name = conf.get('files', 'history_name')
        self.pretrain_name = conf.get('files', 'pretrain_name')
        self.finetune_name = conf.get('files', 'finetune_name')

        # Read Logger Arguments
        self.use_logging = conf.getboolean('neptune', 'use_logging', fallback=None)
        self.neptune_project_id = conf.get('neptune', 'neptune_project_id', fallback=None)
        self.neptune_token_key = conf.get('neptune', 'neptune_token_key', fallback=None)
        self.neptune_api_token = get_env(self.neptune_token_key) if self.neptune_token_key is not None else None

        if self.task == 'corpus':
            self.prepare_parquet = conf.getboolean('extraction', 'prepare_parquet')
            self.max_sequences = conf.getint('extraction', 'max_sequences')

        elif self.task in ['history', 'pre_train', 'fine_tune']:

            self.lr = conf.getfloat('optimization', 'lr')
            self.warmup_proportion = conf.getfloat('optimization', 'warmup_proportion')
            self.weight_decay = conf.getfloat('optimization', 'weight_decay')

            self.max_epochs = conf.getint('train_params', 'max_epochs')
            self.batch_size = conf.getint('train_params', 'batch_size')
            self.max_len_seq = conf.getint('train_params', 'max_len_seq')
            use_gpu = conf.getboolean('train_params', 'use_gpu')
            self.use_pretrained = conf.getboolean('train_params', 'use_pretrained')
            self.save_model = conf.getboolean('train_params', 'save_model')

            self.device = 'cuda' if use_gpu is True else 'cpu'

            self.hidden_size = conf.getint('model_params', 'hidden_size')
            self.layer_dropout = conf.getfloat('model_params', 'layer_dropout')
            self.num_hidden_layers = conf.getint('model_params', 'num_hidden_layers')
            self.num_attention_heads = conf.getint('model_params', 'num_attention_heads')
            self.att_dropout = conf.getfloat('model_params', 'att_dropout')
            self.intermediate_size = conf.getint('model_params', 'intermediate_size')
            self.hidden_act = conf.get('model_params', 'hidden_act')
            self.initializer_range = conf.getfloat('model_params', 'initializer_range')

        elif self.task == 'baseline':
            self.hours = conf.getint('baseline', 'hours')
            self.strategy = conf.get('baseline', 'strategy')
            self.imputation = conf.get('baseline', 'imputation')
            self.scaler = conf.get('baseline', 'scaler')
            self.task = conf.get('baseline', 'task')
            self.feature_select = conf.get('baseline', 'feature_select')
            self.cls = conf.get('baseline', 'cls')
            self.use_saved = conf.getboolean('baseline', 'use_saved')

    def __repr__(self):
        return f'Task: {self.task}'
