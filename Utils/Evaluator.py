import numpy as np
import torch as th
from torchmetrics.classification import AUROC
from torchmetrics.functional.classification import multiclass_f1_score, binary_f1_score


class Evaluator:
    def __init__(
            self,
            task,
            conf,
            scaler=None,
            device='cpu'):
        self.task = task
        self.device = device
        self.conf = conf
        self.scaler = scaler

    def calculate_metrics(self, preds, labs):
        results = {}
        if self.task == 'real':
            results['mae'] = self.mean_absolute_error(preds, labs)
            results['mse'] = self.mean_squared_error(preds, labs)
        elif self.task == 'binary':
            results['f1'] = self.binary_f1(preds, labs)
            results['auc'] = self.auroc_binary(preds, labs)
        elif self.task == 'category':
            results['f1'] = self.category_f1(th.argmax(preds, dim=1), labs.squeeze())
            results['auc'] = self.auroc_multi(preds, labs.squeeze())
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

    def auroc_binary(self, preds, labs):
        metric = AUROC(task='binary').to(self.device)
        return metric(preds, labs).item()

    def auroc_multi(self, preds, labs):
        metric = AUROC(task='multiclass', num_classes=self.conf['num_classes']).to(self.device)
        return metric(preds, labs).item()

    def binary_f1(self, preds, labs):
        thresh = self.fine_tune_threshold(preds, labs, binary_f1_score)
        metric = binary_f1_score(preds, labs, threshold=thresh)
        return metric.item()

    def category_f1(self, preds, labs):
        metric = multiclass_f1_score(preds, labs, num_classes=3)
        return metric.item()

    def mean_absolute_error(self, preds, labs):
        maes = []
        for pred, lab in zip(preds, labs):
            predictions = self.scaler.inverse_transform(pred.mean.to('cpu').numpy())
            labels = self.scaler.inverse_transform(lab.to('cpu').numpy())
            maes.append(np.sum(np.abs(predictions - labels))/len(lab))
        return np.mean(maes)

    def mean_squared_error(self, preds, labs):
        mses = []
        for pred, lab in zip(preds, labs):
            predictions = self.scaler.inverse_transform(pred.mean.to('cpu').numpy())
            labels = self.scaler.inverse_transform(lab.to('cpu').numpy())
            mses.append(np.sum((predictions - labels) ** 2)/len(lab))
        return np.mean(mses)
