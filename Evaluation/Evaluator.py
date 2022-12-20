from torchmetrics.classification import BinaryPrecision
from torchmetrics import MeanAbsoluteError


class Evaluator:
    def __init__(
            self,
            tasks,
            metrics):
        self.tasks = tasks
        self.metrics = metrics

    def calculate_metrics(self, preds, labs):
        results = {}
        for task, metric in zip(self.tasks, self.metrics):
            if task == 'los_real':
                if metric == 'mae':
                    results['los_mae'] = self.mean_absolute_error(preds['los'], labs['los'].unsqueeze(1))
            if task == 'los_binary':
                if metric == 'prec':
                    results['los_prec'] = self.binary_prec(preds['los'], labs['los'].unsqueeze(1))
            if task == 'los_binned':
                pass
            if task == 'req_hosp':
                if metric == 'prec':
                    results['hosp_prec'] = self.binary_prec(preds['hosp'], labs['hosp'].unsqueeze(1))
            if task == 'mortality_30':
                if metric == 'prec':
                    results['m30_prec'] = self.binary_prec(preds['m30'], labs['m30'].unsqueeze(1))

        return results

    def binary_prec(self, preds, labs):
        metric = BinaryPrecision()
        return metric(preds, labs).item()

    def mean_absolute_error(self, preds, labs):
        metric = MeanAbsoluteError()
        return metric(preds, labs).item()
