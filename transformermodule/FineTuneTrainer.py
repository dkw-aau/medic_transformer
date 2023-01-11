from multiprocessing import freeze_support
from transformermodule.Evaluation.Evaluator import Evaluator
from .Model.LengthOfStay import BertForMultiLabelPrediction
from .DataLoader.LOSLoader import LOSLoader
from torch.utils.data import DataLoader
from .Model.utils import BertConfig
from Utils.utils import load_corpus, load_state_dict, save_model_state
from transformermodule.utils import get_model_config
from .Model.optimiser import adam
from tqdm import tqdm
import torch as th
import warnings
import torch
import time
import os


class FineTuneTrainer:
    def __init__(
            self,
            args):

        self.args = args
        freeze_support()
        warnings.filterwarnings(action='ignore')

        # Load Corpus
        corpus = load_corpus(os.path.join(args.path['data_fold'], args.corpus_name))
        corpus = corpus.get_subset_by_min_hours(min_hours=24)
        corpus.cut_sequences_by_hours(hours=24)
        corpus.substract_los_hours(hours=24)
        train, evalu, test = corpus.get_data_split()

        vocab = corpus.get_vocabulary()
        print(f'Vocabulary Size: {len(vocab["token2index"])}')

        tasks = ['los_category']
        # TODO: Implement AUC-ROC, AUC-PR, KAPPA, print confusion matrix
        metrics = ['acc']
        conf = {
            'los_binary_threshold': 5,
            'classes': [1, 2, 3, 4, 5, 6, 7, 14]
        }

        self.evaluator = Evaluator(tasks=tasks, metrics=metrics, device=self.args.device, conf=conf)

        # Train
        Dset = LOSLoader(token2idx=vocab['token2index'], sequences=train, max_len=args.max_len_seq, conf=conf, tasks=tasks)
        self.trainloader = DataLoader(dataset=Dset, batch_size=args.batch_size, shuffle=True, num_workers=0)

        # Eval
        Dset = LOSLoader(token2idx=vocab['token2index'], sequences=evalu, max_len=args.max_len_seq, conf=conf, tasks=tasks)
        self.evaluloader = DataLoader(dataset=Dset, batch_size=args.batch_size, shuffle=True, num_workers=0)

        # Test
        Dset = LOSLoader(token2idx=vocab['token2index'], sequences=test, max_len=args.max_len_seq, conf=conf, tasks=tasks)
        self.testloader = DataLoader(dataset=Dset, batch_size=args.batch_size, shuffle=False, num_workers=0)

        self.evaluator.baseline_results(self.trainloader, self.evaluloader, self.testloader)

        model_config = get_model_config(vocab, args)
        feature_dict = {
            'word': True,
            'position': True,
            'seg': True
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
        self.optim = adam(params=list(model.named_parameters()), args=args)

    def train(self, epochs):
        for e in range(0, epochs):
            self.epoch(e)
            loss, metrics = self.evaluation()
            metric_string = ' - '.join([f'{key}: {round(value, 3)}' for key, value in metrics.items()])
            print(f'Loss: {round(loss, 3)} - {metric_string}')

            save_model_state(self.model, self.args.path['out_fold'], self.args.finetune_name)

    def epoch(self, e):
        self.model.train()
        tr_loss = 0
        tr_metrics = {}
        epoch_time = time.time()

        loader_iter = tqdm(self.trainloader, ncols=120, position=0)
        for step, batch in enumerate(loader_iter, 1):
            input_ids, posi_ids, seg_ids, att_mask, targets, pat_ids = batch

            input_ids = input_ids.to(self.args.device)
            posi_ids = posi_ids.to(self.args.device)
            seg_ids = seg_ids.to(self.args.device)
            att_mask = att_mask.to(self.args.device)
            targets = {key: lab.to(self.args.device) for key, lab in targets.items()}

            loss, logits = self.model(input_ids, posi_ids, seg_ids, targets=targets, attention_mask=att_mask)

            loss.backward()

            new_metrics = self.evaluator.calculate_metrics(logits, targets)
            tr_metrics = {k: tr_metrics.get(k, 0) + new_metrics.get(k, 0) for k in set(tr_metrics) | set(new_metrics)}
            disp_metrics = {key: value / step for key, value in tr_metrics.items()}

            tr_loss += loss.item()
            print_dict = {'epoch': e, 'loss': tr_loss / step}
            print_dict.update(disp_metrics)
            loader_iter.set_postfix(print_dict)

            self.optim.step()
            self.optim.zero_grad()

        return tr_loss / step, time.time() - epoch_time

    def evaluation(self):
        self.model.eval()
        y_preds = {}
        y_label = {}
        tr_loss = 0
        for step, (input_ids, posi_ids, seg_ids, att_mask, targets, pat_ids) in enumerate(self.testloader, 1):
            self.model.eval()

            input_ids = input_ids.to(self.args.device)
            posi_ids = posi_ids.to(self.args.device)
            seg_ids = seg_ids.to(self.args.device)
            att_mask = att_mask.to(self.args.device)
            targets = {key: lab.to(self.args.device) for key, lab in targets.items()}

            with torch.no_grad():
                loss, logits = self.model(input_ids, posi_ids, seg_ids, targets=targets, attention_mask=att_mask)

            tr_loss += loss.item()

            for key, value in logits.items():
                y_preds[key] = value if key not in y_preds else th.cat((y_preds[key], logits[key]))
            for key, value in targets.items():
                y_label[key] = value if key not in y_label else th.cat((y_label[key], targets[key]))

        metrics = self.evaluator.calculate_metrics(y_preds, y_label)

        return tr_loss / step, metrics
