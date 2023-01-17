import numpy as np
from torch.utils.data import DataLoader

from Utils.EarlyStopping import EarlyStopping
from .DataLoader.HistoryLoader import HistoryLoader
import pytorch_pretrained_bert as Bert
from transformermodule.Evaluation.Evaluator import Evaluator
import torch as th
from .Model.LengthOfStay import BertForMultiLabelPrediction
from .utils import get_model_config
from .Model.utils import BertConfig
from .Model.optimiser import adam
from Utils.utils import load_corpus, save_model_state, create_folder, load_state_dict
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import warnings
import time
import os


class HistoryTrainer:
    def __init__(
            self,
             args):

        self.args = args
        self.logger = self.args.logger
        warnings.filterwarnings(action='ignore')
        create_folder(args.path['out_fold'])

        # TODO: Implement AUC-ROC, AUC-PR, KAPPA, print confusion matrix
        self.conf = {
            'task': 'category',
            'metrics': ['f1', 'auc'],
            'binary_thresh': 2,
            'cats': [2, 7],
            'years': [2018, 2019, 2020, 2021],
            'types': ['apriori', 'adm', 'proc', 'vital', 'lab'],  # 'apriori', 'vital', 'diag', 'apriori', 'adm', 'proc', 'lab'
            'max_hours': 24
        }

        # Load corpus
        corpus = load_corpus(os.path.join(args.path['data_fold'], args.corpus_name))

        vocab = corpus.prepare_corpus(
            self.conf
        )
        corpus.create_pos_ids(event_dist=300)
        corpus.create_train_evel_test_idx(train_size=0.8)

        # Select subset corpus
        train_x, evalu_x, test_x = corpus.split_train_eval_test()

        # Setup Evaluator
        self.evaluator = Evaluator(conf=self.conf, device=self.args.device)

        # Setup dataloaders
        Dset = HistoryLoader(token2idx=vocab['token2index'], sequences=train_x, max_len=args.max_len_seq, conf=self.conf)
        self.trainloader = DataLoader(dataset=Dset, batch_size=args.batch_size, shuffle=True, num_workers=0)

        Dset = HistoryLoader(token2idx=vocab['token2index'], sequences=evalu_x, max_len=args.max_len_seq, conf=self.conf)
        self.evalloader = DataLoader(dataset=Dset, batch_size=args.batch_size, shuffle=True, num_workers=0)

        Dset = HistoryLoader(token2idx=vocab['token2index'], sequences=test_x, max_len=args.max_len_seq, conf=self.conf)
        self.testloader = DataLoader(dataset=Dset, batch_size=args.batch_size, shuffle=True, num_workers=0)

        # Create Bert Model
        model_config = get_model_config(vocab, args)
        feature_dict = {
            'word': True,
            'position': True,
            'age': True,
            'gender': True,
            'seg': False
        }
        bert_conf = BertConfig(model_config)

        # Setup Model
        model = BertForMultiLabelPrediction(
            args=args,
            bert_conf=bert_conf,
            feature_dict=feature_dict,
            cls_conf=self.conf,
            class_weights=None)

        self.model = model.to(args.device)

        # Initialize model parameters
        #if args.use_pretrained:
        #    print(f'Loading state model with name: {args.load_name}')
        #    self.model = load_state_dict(os.path.join(args.path['out_fold'], args.load_name), self.model)

        self.optim = adam(params=list(self.model.named_parameters()), args=args)

    def train(self, epochs):

        self.logger.start_log()

        self.logger.log_value('Experiment', self.args.experiment_name)
        self.logger.log_value('threshold', self.conf['binary_thresh'])
        self.logger.log_value('categories', self.conf['cats'])

        stopper = EarlyStopping(self.args.patience, f'{os.path.join(self.args.path["out_fold"], self.conf["task"])}.pt')

        for e in range(0, epochs):
            train_loss, train_metrics = self.evaluation(self.trainloader)
            eval_loss, eval_metrics = self.evaluation(self.evalloader)
            train_metrics.update({'loss': train_loss})
            eval_metrics.update({'loss': eval_loss})

            self.logger.log_metrics(train_metrics, 'train')
            self.logger.log_metrics(eval_metrics, 'eval')

            self.logger.report_metrics(train_metrics, 'train')
            self.logger.report_metrics(eval_metrics, 'eval')

            if stopper.step(eval_loss, self.model):
                self.logger.info('Early Stop!\tEpoch:' + str(e))
                break

            self.epoch(e)

        print('Evaluating trained model')
        self.model = stopper.load_model(self.model)

        test_loss, test_metrics = self.evaluation(self.testloader)
        test_metrics.update({'loss': test_loss})

        self.logger.report_metrics(test_metrics, 'test')
        self.logger.log_value('test_loss', test_loss)
        self.logger.log_values(test_metrics, 'test')

        self.logger.stop_log()

    def epoch(self, e):
        self.model.train()
        tr_loss = 0
        epoch_time = time.time()

        loader_iter = tqdm(self.trainloader, ncols=120, position=0)
        for step, batch in enumerate(loader_iter, 1):
            input_ids, posi_ids, age_ids, gender_ids, att_mask, labels, pat_ids = batch

            input_ids = input_ids.to(self.args.device)
            posi_ids = posi_ids.to(self.args.device)
            age_ids = age_ids.to(self.args.device)
            gender_ids = gender_ids.to(self.args.device)
            att_mask = att_mask.to(self.args.device)
            labels = labels.to(self.args.device)

            loss, logits = self.model(input_ids, posi_ids, age_ids, gender_ids, targets=labels, attention_mask=att_mask)

            loss.backward()

            tr_loss += loss.item()
            print_dict = {'epoch': e, 'loss': tr_loss / step}
            loader_iter.set_postfix(print_dict)

            self.optim.step()
            self.optim.zero_grad()

        return tr_loss / step, time.time() - epoch_time

    def evaluation(self, loader):
        self.model.eval()
        tr_loss = 0
        y_preds, y_label = None, None
        for step, (input_ids, posi_ids, age_ids, gender_ids, att_mask, labels, pat_ids) in enumerate(loader, 1):
            self.model.eval()

            input_ids = input_ids.to(self.args.device)
            posi_ids = posi_ids.to(self.args.device)
            age_ids = age_ids.to(self.args.device)
            gender_ids = gender_ids.to(self.args.device)
            att_mask = att_mask.to(self.args.device)
            labels = labels.to(self.args.device)

            with th.no_grad():
                loss, logits = self.model(input_ids, posi_ids, age_ids, gender_ids, targets=labels, attention_mask=att_mask)

            tr_loss += loss.item()

            y_preds = logits if y_preds is None else th.cat((y_preds, logits))
            y_label = labels if y_label is None else th.cat((y_label, labels))

        metrics = self.evaluator.calculate_metrics(y_preds, y_label)
        #print(self.evaluator.confusion_matrix(y_preds, y_label))

        return tr_loss / step, metrics
