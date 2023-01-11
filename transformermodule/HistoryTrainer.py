import numpy as np
from torch.utils.data import DataLoader
from .DataLoader.HistoryLoader import HistoryLoader
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

        # Load corpus
        corpus = load_corpus(os.path.join(args.path['data_fold'], args.corpus_name))
        corpus.create_train_evel_test_idx(train_size=0.8)
        corpus = corpus.get_subset_corpus(min_hours=24)
        train, evalu, test = corpus.split_train_eval_test()
        vocab = corpus.get_vocabulary()

        # TODO: Implement AUC-ROC, AUC-PR, KAPPA, print confusion matrix
        tasks = ['los_category']
        metrics = ['acc']
        conf = {
            'los_binary_threshold': 5,
            'classes': [1, 2, 3, 4, 5, 6, 7, 14]
        }

        self.evaluator = Evaluator(tasks=tasks, metrics=metrics, device=self.args.device, conf=conf)

        # Setup dataloader
        Dset = HistoryLoader(token2idx=vocab['token2index'], sequences=train, max_len=args.max_len_seq, conf=conf, tasks=tasks)
        self.trainloader = DataLoader(dataset=Dset, batch_size=args.batch_size, shuffle=True, num_workers=0)

        # Eval
        Dset = HistoryLoader(token2idx=vocab['token2index'], sequences=evalu, max_len=args.max_len_seq, conf=conf, tasks=tasks)
        self.evalloader = DataLoader(dataset=Dset, batch_size=args.batch_size, shuffle=True, num_workers=0)

        # Compute Class Weights
        # TODO: Move this to other function
        #y_label = {}
        #for input_ids, att_mask, labels, pat_ids in self.trainloader:
        #    for key, value in labels.items():
        #        y_label[key] = value if key not in y_label else th.cat((y_label[key], labels[key]))

        #class_weights = compute_class_weight('balanced', classes=np.unique(y_label['los_category'].squeeze()), y=y_label['los_category'].squeeze().numpy())
        #class_weights = th.tensor(class_weights, dtype=th.float)

        # Create Bert Model
        model_config = get_model_config(vocab, args)
        feature_dict = {
            'word': True,
            'position': True,
            'seg': False
        }
        bert_conf = BertConfig(model_config)

        # Setup Model
        model = BertForMultiLabelPrediction(
            args=args,
            config=bert_conf,
            feature_dict=feature_dict,
            cls_heads=tasks,
            cls_config=conf)

        self.model = model.to(args.device)

        # Initialize model parameters
        if args.use_pretrained:
            print(f'Loading state model with name: {args.finetune_name}')
            self.model = load_state_dict(os.path.join(args.path['out_fold'], args.finetune_name), self.model)

        self.optim = adam(params=list(model.named_parameters()), args=args)

    def train(self, epochs):

        self.logger.start_log()

        for e in range(0, epochs):
            self.epoch(e)
            train_loss, train_metrics = self.evaluation(self.trainloader)
            eval_loss, eval_metrics = self.evaluation(self.evalloader)
            self.logger.log_sequence('train/loss', train_loss)
            self.logger.log_sequence('eval/loss', eval_loss)
            train_metric_string = ' - '.join([f'{key}: {round(value, 3)}' for key, value in train_metrics.items()])
            eval_metric_string = ' - '.join([f'{key}: {round(value, 3)}' for key, value in eval_metrics.items()])
            print(f'Train: Loss {round(train_loss, 3)}, {train_metric_string}')
            print(f'Eval: Loss {round(eval_loss, 3)}, {eval_metric_string}')
            #save_model_state(self.model, self.args.path['out_fold'], self.args.finetune_name)

    def epoch(self, e):
        self.model.train()
        tr_loss = 0
        epoch_time = time.time()

        loader_iter = tqdm(self.trainloader, ncols=120, position=0)
        for step, batch in enumerate(loader_iter, 1):
            input_ids, posi_ids, att_mask, labels, pat_ids = batch

            input_ids = input_ids.to(self.args.device)
            posi_ids = posi_ids.to(self.args.device)
            att_mask = att_mask.to(self.args.device)
            labels = {key: lab.to(self.args.device) for key, lab in labels.items()}

            loss, logits = self.model(input_ids, posi_ids, targets=labels, attention_mask=att_mask)

            loss.backward()

            tr_loss += loss.item()
            print_dict = {'epoch': e, 'loss': tr_loss / step}
            loader_iter.set_postfix(print_dict)

            self.optim.step()
            self.optim.zero_grad()

        return tr_loss / step, time.time() - epoch_time

    def evaluation(self, loader):
        self.model.eval()
        y_preds = {}
        y_label = {}
        tr_loss = 0
        for step, (input_ids, posi_ids, att_mask, labels, pat_ids) in enumerate(loader, 1):
            self.model.eval()

            input_ids = input_ids.to(self.args.device)
            posi_ids = posi_ids.to(self.args.device)
            att_mask = att_mask.to(self.args.device)
            labels = {key: lab.to(self.args.device) for key, lab in labels.items()}

            with th.no_grad():
                loss, logits = self.model(input_ids, posi_ids, targets=labels, attention_mask=att_mask)

            tr_loss += loss.item()

            for key, value in logits.items():
                y_preds[key] = value if key not in y_preds else th.cat((y_preds[key], logits[key]))
            for key, value in labels.items():
                y_label[key] = value if key not in y_label else th.cat((y_label[key], labels[key]))

        metrics = self.evaluator.calculate_metrics(y_preds, y_label)
        #print(self.evaluator.confusion_matrix(y_preds, y_label))

        return tr_loss / step, metrics
