import torch as th
import torch.nn
from tqdm import tqdm
from Models.BaseModel import BaseModel
from torchmetrics.functional import auc
from torchmetrics.functional import average_precision
from torchmetrics.functional import mean_absolute_error
from torchmetrics.functional import f1_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import numpy as np
import shap


class Experiment:
    def __init__(self, args, model, target='binary'):
        if target == 'binary':
            self.loss = torch.nn.BCELoss()
            self.metric = metrics.roc_auc_score
            self.metric_name = 'ROC'
        elif target == 'regression':
            self.loss = torch.nn.L1Loss()
            self.metric = mean_absolute_error
            self.metric_name = 'mae'
        elif target == 'groups':
            self.loss = torch.nn.CrossEntropyLoss()
            self.metric = metrics.roc_auc_score
            self.metric_name = 'ROC'

        self.train_loader = None
        self.test_loader = None

        self.model = model
        self.epochs = args['epochs']
        self.device = args['device']

        self.lr = args['lr']
        self.weight_decay = args['weight_decay']

        self.opt = th.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def train(self, train_loader, test_loader):

        self.train_loader, self.test_loader = train_loader, test_loader

        # Set early stopping criteria
        print(f'Starting Training')

        ncols = 120
        epoch_iter = tqdm(range(self.epochs), ncols=ncols, position=0)
        for epoch in epoch_iter:
            self.train_step()
            metric_dict, losses = self._evaluate(['train', 'test'])
            epoch_iter.set_postfix({
                self.metric_name: ', '.join([str(round(val, 3)) for val in metric_dict.values()]),
                'loss': ', '.join([str(round(val, 3)) for val in losses.values()])})

        return self.model

    def _evaluate(self, modes):
        metrics = {}
        losses = {}
        for mode in modes:
            metrics[mode], losses[mode] = self.test_step(mode)

        return metrics, losses

    def train_step(self):
        loss_all = 0.0
        for i, (inputs, labels, idx) in enumerate(self.train_loader, 1):
            preds = self.model(inputs)

            self.opt.zero_grad()

            loss = self.loss(preds, labels)

            loss.backward()
            self.opt.step()

            loss_all += loss.item()

        return loss_all / i

    def test_step(self, mode: str) -> (dict, float):
        self.model.eval()

        loader = None
        if mode == "train":
            loader = self.train_loader
        elif mode == "test":
            loader = self.test_loader

        with torch.no_grad():
            loss_all = 0.0
            all_labs = torch.Tensor()
            all_preds = torch.Tensor()
            for i, (inputs, labels, idx) in enumerate(loader, 1):
                with torch.no_grad():
                    preds = self.model(inputs)

                loss_all += self.loss(preds, labels).item()

                all_preds = th.cat((all_preds, preds), 0)
                all_labs = th.cat((all_labs, labels), 0)

            score = self.metric(all_labs, all_preds)
            return score.item(), loss_all / i

    def get_predictions(self, mode: str) -> (list, list):
        loader = None
        if mode == "train":
            loader = self.train_loader
        elif mode == "test":
            loader = self.test_loader

        with torch.no_grad():
            all_labs = []
            all_preds = []
            for i, (inputs, labels, idx) in enumerate(loader, 1):
                with torch.no_grad():
                    preds = self.model(inputs)

                all_preds.extend(preds.numpy().flatten())
                all_labs.extend(labels.numpy().flatten())

        return all_preds, all_labs
