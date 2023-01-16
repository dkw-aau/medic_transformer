from torchmetrics.classification import BinaryPrecision, BinaryF1Score, BinaryAccuracy, MulticlassAccuracy, AUROC
from torchmetrics.functional import accuracy, f1_score as tf1_score
from torchmetrics.functional.classification import multiclass_f1_score, binary_f1_score
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
        self.metrics = conf['metrics']
        self.device = device
        self.conf = conf

    def calculate_metrics(self, preds, labs):
        results = {}
        for metric in self.metrics:
            if self.task == 'real':
                if metric == 'mae':
                    results[metric] = self.mean_absolute_error(preds, labs)
            elif self.task in ('binary', 'm30'):
                if metric == 'acc':
                    results[metric] = self.binary_acc(preds, labs)
                if metric == 'f1':
                    results[metric] = self.binary_f1(preds, labs)
                if metric == 'auc':
                    results[metric] = self.auroc_binary(preds, labs)
            elif self.task == 'category':
                if metric == 'acc':
                    results[metric] = self.multi_acc(th.argmax(preds, dim=1), labs.squeeze())
                if metric == 'auc':
                    results[metric] = self.auroc_multiclass(preds, labs.squeeze())
                if metric == 'f1':
                    results[metric] = self.category_f1(th.argmax(preds, dim=1), labs.squeeze())
            else:
                exit(f'Task: {self.task} not implemented')

        return results

    def fine_tune_threshold(self, preds, labels, function, named_args=None):
        if named_args is None:
            named_args = {}
        best_threshold, best_value = 0, 0

        for threshold in range(0, 100):
            thresh = threshold / 100
            new_value = function(preds, labels, threshold=thresh, **named_args).item()

            if new_value > best_value:
                best_value = new_value
                best_threshold = thresh

        return best_threshold

    def binary_acc(self, preds, labs):
        thresh = self.fine_tune_threshold(preds, labs, accuracy)
        metric = accuracy(preds, labs, threshold=thresh, num_classes=2, task='binary')
        return metric.item()

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
        thresh = self.fine_tune_threshold(preds, labs, binary_f1_score)
        metric = tf1_score(preds, labs, threshold=thresh, num_classes=2, task='binary')
        return metric.item()

    def category_f1(self, preds, labs):
        metric = multiclass_f1_score(preds, labs, num_classes=3)
        return metric.item()

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
    """
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

            for metric in self.metrics:
                if metric == 'acc':
                    print(f'Baseline Prec for Train: {round(accuracy_score(train_y, train_preds), 2)}')
                    print(f'Baseline Prec for Eval: {round(accuracy_score(evalu_y, evalu_preds), 2)}')
                    print(f'Baseline Prec for Test: {round(accuracy_score(test_y, test_preds), 2)}')
                if metric == 'f1':
                    print(f'Baseline F1 for Train: {round(f1_score(train_y, train_preds, average="macro"), 2)}')
                    print(f'Baseline F1 for Eval: {round(f1_score(evalu_y, evalu_preds, average="macro"), 2)}')
                    print(f'Baseline F1 for Test: {round(f1_score(test_y, test_preds, average="macro"), 2)}')
                if metric == 'auc':
                    if self.task == 'binary':
                        pass
                    elif self.task == 'category':
                        train_preds = np.eye(len(self.conf['cats']) + 1)[train_preds]
                        evalu_preds = np.eye(len(self.conf['cats']) + 1)[evalu_preds]
                        test_preds = np.eye(len(self.conf['cats']) + 1)[test_preds]
                    print(f'Baseline AUC for Train: {round(roc_auc_score(train_y, train_preds, multi_class="ovo"), 2)}')
                    print(f'Baseline AUC for Eval: {round(roc_auc_score(evalu_y, evalu_preds, multi_class="ovo"), 2)}')
                    print(f'Baseline AUC for Test: {round(roc_auc_score(test_y, test_preds, multi_class="ovo"), 2)}')
    """
