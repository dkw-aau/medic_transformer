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

        # Data paths
        self.path = {
            'data_fold': path + '/Data/',
            'out_fold': path + '/Outputs/',
        }

        # Experiment settings
        self.workload = conf.get('experiment', 'workload')
        self.task = conf.get('experiment', 'task')
        self.corpus = conf.get('experiment', 'corpus')
        self.binary_thresh = conf.getint('experiment', 'binary_thresh')
        self.categories = [int(x) for x in conf.get('experiment', 'categories').split(',')]
        self.years = [int(x) for x in conf.get('experiment', 'years').split(',')]
        self.types = [str(x) for x in conf.get('experiment', 'types').split(',')]
        self.seq_hours = conf.getint('experiment', 'seq_hours')
        self.clip_los = conf.getint('experiment', 'clip_los')

        self.load_mlm = conf.getboolean('experiment', 'load_mlm')
        self.save_model = conf.getboolean('experiment', 'save_model')

        # Read Logger Arguments
        self.use_logging = conf.getboolean('logging', 'use_logging', fallback=None)
        self.neptune_project_id = conf.get('logging', 'neptune_project_id', fallback=None)
        self.neptune_token_key = conf.get('logging', 'neptune_token_key', fallback=None)
        self.neptune_api_token = get_env(self.neptune_token_key) if self.neptune_token_key is not None else None
        self.experiment_name = conf.get('logging', 'experiment_name')

        # Data transform config
        self.prepare_parquet = conf.getboolean('extraction', 'prepare_parquet')
        self.max_sequences = conf.getint('extraction', 'max_sequences')

        # Optimization
        self.lr = conf.getfloat('optimization', 'lr')
        self.warmup_proportion = conf.getfloat('optimization', 'warmup_proportion')
        self.weight_decay = conf.getfloat('optimization', 'weight_decay')

        # Training config
        self.epochs = conf.getint('train_params', 'epochs')
        self.batch_size = conf.getint('train_params', 'batch_size')
        self.max_len_seq = conf.getint('train_params', 'max_len_seq')
        use_gpu = conf.getboolean('train_params', 'use_gpu')
        self.patience = conf.getint('train_params', 'patience')
        self.device = 'cuda' if use_gpu is True else 'cpu'

        # Model parameters
        self.hidden_size = conf.getint('model_params', 'hidden_size')
        self.layer_dropout = conf.getfloat('model_params', 'layer_dropout')
        self.num_hidden_layers = conf.getint('model_params', 'num_hidden_layers')
        self.num_attention_heads = conf.getint('model_params', 'num_attention_heads')
        self.att_dropout = conf.getfloat('model_params', 'att_dropout')
        self.intermediate_size = conf.getint('model_params', 'intermediate_size')
        self.hidden_act = conf.get('model_params', 'hidden_act')
        self.initializer_range = conf.getfloat('model_params', 'initializer_range')
        self.features = [str(x) for x in conf.get('model_params', 'features').split(',')]

        # Baseline parameters
        self.hours = conf.getint('baseline', 'hours')
        self.strategy = conf.get('baseline', 'strategy')
        self.imputation = conf.get('baseline', 'imputation')
        self.scaler = conf.get('baseline', 'scaler')
        self.feature_select = conf.get('baseline', 'feature_select')
        self.cls = conf.get('baseline', 'cls')
        self.use_saved = conf.getboolean('baseline', 'use_saved')

    def __repr__(self):
        return f'Task: {self.task}'
