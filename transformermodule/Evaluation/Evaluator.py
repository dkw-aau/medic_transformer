from torchmetrics.classification import BinaryPrecision, BinaryF1Score, BinaryAccuracy, MulticlassAccuracy, AUROC
from sklearn.metrics import confusion_matrix, roc_auc_score
from torchmetrics import MeanAbsoluteError
from sklearn.metrics import precision_score, accuracy_score, f1_score
from sklearn.dummy import DummyClassifier
import numpy as np
import torch as th
import random


class Evaluator:
    def __init__(
            self,
            conf=None,
            device='cpu'):
        self.task = conf['task']
        self.metric = conf['metric']
        self.device = device
        self.conf = conf

    def calculate_metrics(self, preds, labs):
        result = None
        if self.task == 'real':
            if self.metric == 'mae':
                result = self.mean_absolute_error(preds, labs)
        elif self.task in ('binary', 'm30'):
            if self.metric == 'acc':
                result = self.binary_acc(preds, labs)
            if self.metric == 'f1':
                result = self.binary_f1(preds, labs)
            if self.metric == 'auc':
                result = self.auroc_binary(preds, labs)
        elif self.task == 'category':
            if self.metric == 'acc':
                result = self.multi_acc(th.argmax(preds, dim=1), labs.squeeze())
            if self.metric == 'auc':
                result = self.auroc_multiclass(preds, labs.squeeze())
        else:
            exit(f'Task: {self.task} not implemented')

        return result

    def binary_acc(self, preds, labs):
        metric = BinaryAccuracy().to(self.device)
        return metric(preds, labs).item()

    def auroc_binary(self, preds, labs):
        metric = AUROC(task='binary').to(self.device)
        return metric(preds, labs).item()

    def auroc_multiclass(self, preds, labs):
        metric = AUROC(task='multiclass', num_classes=len(self.conf['cats']) + 1).to(self.device)
        return metric(preds, labs).item()

    def multi_acc(self, preds, labs):
        metric = MulticlassAccuracy(len(self.conf['cats']) + 1).to(self.device)
        return metric(preds, labs).item()

    def binary_f1(self, preds, labs):
        metric = BinaryF1Score().to(self.device)
        return metric(preds, labs).item()

    def mean_absolute_error(self, preds, labs):
        metric = MeanAbsoluteError().to(self.device)
        return metric(preds, labs).item()

    def get_max_mae(self, sequences):
        losses = [seq.length_of_stay for seq in sequences]
        mean_los = sum(losses) / len(losses)

        mae = sum([abs(los - mean_los) for los in losses]) / len(losses)

        return mae

    def get_max_accuracy(self, labels):
        cls1_accuracy = accuracy_score(labels, [1] * len(labels))
        cls2_accuracy = accuracy_score(labels, [0] * len(labels))
        return max(cls1_accuracy, cls2_accuracy)

    def get_max_f1(self, labels):
        cls1_f1 = f1_score(labels, [1] * len(labels))
        cls2_f1 = f1_score(labels, [0] * len(labels))
        return max(cls1_f1, cls2_f1)

    def confusion_matrix(self, preds, labs):
        labs = labs.squeeze().cpu().numpy()
        if self.task == 'category':
            preds = th.argmax(preds, dim=1).cpu().numpy()
        elif self.task in ['binary', 'm30']:
            preds = preds.squeeze().cpu().numpy()
            preds[preds >= 0.5] = 1
            preds[preds < 0.5] = 0

        return confusion_matrix(labs, preds)

    def get_loader_labels(self, loader, task):
        if task in ['binary', 'category', 'm30']:
            data_y = [targets[task].tolist() for _, _, _, _, targets, _ in loader]
            data_y = [item[0] for sublist in data_y for item in sublist]

        return data_y

    def get_dummy_preds(self, labels):
        dummy_clf = DummyClassifier(strategy="most_frequent")
        random_input = [random.randrange(1, 50, 1) for _ in range(len(labels))]
        clf = dummy_clf.fit(random_input, labels)
        preds = clf.predict(random_input)

        return preds

    def baseline_results(self, train_x, train_y, evalu_x, evalu_y, test_x, test_y):
        if self.task == 'real':
            if self.metric == 'mae':
                print(f'Baseline Mean MAE for Train: {round(self.get_max_mae(train_x), 3)}')
                print(f'Baseline Mean MAE for Eval: {round(self.get_max_mae(evalu_x), 3)}')
                print(f'Baseline Mean MAE for Test: {round(self.get_max_mae(test_x), 3)}')

        if self.task in ['binary', 'category', 'm30']:
            train_preds = self.get_dummy_preds(train_y)
            evalu_preds = self.get_dummy_preds(evalu_y)
            test_preds = self.get_dummy_preds(test_y)

            if self.metric == 'acc':
                print(f'Baseline Prec for Train: {round(accuracy_score(train_y, train_preds), 2)}')
                print(f'Baseline Prec for Eval: {round(accuracy_score(evalu_y, evalu_preds), 2)}')
                print(f'Baseline Prec for Test: {round(accuracy_score(test_y, test_preds), 2)}')
            if self.metric == 'f1':
                print(f'Baseline F1 for Train: {round(f1_score(train_y, train_preds), 2)}')
                print(f'Baseline F1 for Eval: {round(f1_score(evalu_y, evalu_preds), 2)}')
                print(f'Baseline F1 for Test: {round(f1_score(test_y, test_preds), 2)}')
            if self.metric == 'auc':
                train_preds = np.eye(len(self.conf['cats']) + 1)[train_preds]
                evalu_preds = np.eye(len(self.conf['cats']) + 1)[evalu_preds]
                test_preds = np.eye(len(self.conf['cats']) + 1)[test_preds]
                print(f'Baseline F1 for Train: {round(roc_auc_score(train_y, train_preds, multi_class="ovo"), 2)}')
                print(f'Baseline F1 for Eval: {round(roc_auc_score(evalu_y, evalu_preds, multi_class="ovo"), 2)}')
                print(f'Baseline F1 for Test: {round(roc_auc_score(test_y, test_preds, multi_class="ovo"), 2)}')
