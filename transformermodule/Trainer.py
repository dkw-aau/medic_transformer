import os
import warnings

import pytorch_pretrained_bert as Bert
from torch.utils.data import DataLoader

from Utils.Corpus import Corpus
from Utils.EarlyStopping import EarlyStopping
from Utils.Evaluator import Evaluator
from Utils.Scaler import Scaler
from Utils.logger import Logger
from Utils.utils import get_model_config, BertConfig
from Utils.utils import load_state_dict
from .DataLoader.LOSLoader import LOSLoader
from .DataLoader.MLMLoader import MLMLoader
from .Model.MBERT import MBERT


class Trainer:
    def __init__(self, args):
        self.args = args
        self.path = args.path
        self.task = args.task
        self.device = args.device
        self.epochs = args.epochs
        self.load_mlm = args.load_mlm
        self.workload = args.workload
        self.features = args.features
        self.batch_size = args.batch_size
        self.max_len_seq = args.max_len_seq
        self.binary_thresh = args.binary_thresh
        self.num_classes = len(self.args.categories) + 1

        warnings.filterwarnings("ignore")

        # Create Logger
        self.logger = Logger(args)
        self.logger.start_log()
        self.logger.log_value('Experiment', self.args.experiment_name)

        # Model configuration
        self.model_config = {
            'hidden_size': args.hidden_size,
            'max_len_seq': args.max_len_seq,
            'layer_dropout': args.layer_dropout,
            'num_hidden_layers': args.num_hidden_layers,
            'num_attention_heads': args.num_attention_heads,
            'att_dropout': args.att_dropout,
            'intermediate_size': args.intermediate_size,
            'hidden_act': args.hidden_act,
            'initializer_range': args.initializer_range
        }

        # Corpus configuration
        self.corpus_conf = {
            'task': args.task,
            'binary_thresh': args.binary_thresh,
            'cats': args.categories,
            'years': args.years,
            'types': args.types,
            'seq_hours': args.seq_hours,
            'clip_los': args.clip_los
        }

        self.evaluator_conf = {
            'num_classes': self.num_classes
        }

        # Prepare corpus
        self.corpus = Corpus(
            data_path=args.path['data_fold'],
            file_names=args.file_names
        )
        self.vocab = self.corpus.prepare_corpus(self.corpus_conf)

        # Split data
        train, evalu, test = self.corpus.split_train_eval_test()
        self.scaler = self.get_scaler()
        self.train_loader = self.get_data_loader(train)
        self.evalu_loader = self.get_data_loader(evalu)
        self.test_loader = self.get_data_loader(test)

        # Setup Evaluator
        self.evaluator = Evaluator(
            task=self.task,
            conf=self.evaluator_conf,
            scaler=self.scaler,
            device=self.device)

        # Create M-BERT model
        self.model = self.get_model().to(self.device)

        # Setup Stopper
        self.stopper = self.get_stopper()

        # Initialize model from mlm
        self.preload_mlm()

        # Initialize optimizer
        self.optim = self.adam(params=list(self.model.named_parameters()), args=args)

    def get_stopper(self):
        return EarlyStopping(
            patience=self.args.patience,
            save_path=f'{os.path.join(self.path["out_fold"], self.task)}.pt',
            save_trained=self.args.save_model
        )

    def preload_mlm(self):
        if self.load_mlm:
            print(f'Loading mlm state model')
            self.model = load_state_dict(os.path.join(self.path['out_fold'], 'mlm.pt'), self.model)

    def get_model(self):
        # Model configuration
        bert_conf = get_model_config(self.vocab, self.model_config)
        bert_conf = BertConfig(bert_conf)

        # Create M-BERT Model
        return MBERT(
            bert_conf=bert_conf,
            workload=self.workload,
            features=self.features,
            task=self.task,
            scaler=self.scaler,
            num_classes=self.num_classes
        )

    def get_scaler(self):
        # Create scaler for real value predictions
        if self.task == 'real':
            all_labs = [seq.label for seq in self.corpus.sequences]
            return Scaler('box-cox').fit(all_labs)
        else:
            return None

    def get_data_loader(self, data):
        if self.workload == 'mlm':
            Dset = MLMLoader(
                token2idx=self.vocab['token2index'],
                sequences=data,
                max_len=self.max_len_seq)
        else:
            Dset = LOSLoader(
                token2idx=self.vocab['token2index'],
                sequences=data,
                max_len=self.max_len_seq,
                task=self.task,
                scaler=self.scaler)

        return DataLoader(
            dataset=Dset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0)

    def adam(self, params, args):
        config = {
            'lr': args.lr,
            'warmup_proportion': args.warmup_proportion,
            'weight_decay': args.weight_decay
        }

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        optimizer_grouped_parameters = [
            {'params': [p for n, p in params if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in params if any(nd in n for nd in no_decay)], 'weight_decay': 0}
        ]

        optim = Bert.optimization.BertAdam(
            optimizer_grouped_parameters,
            lr=config['lr'],
            warmup=config['warmup_proportion']
        )

        return optim