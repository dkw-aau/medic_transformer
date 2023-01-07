from torchmetrics.classification import BinaryPrecision, BinaryF1Score, BinaryAccuracy, MulticlassAccuracy
from torchmetrics import MeanAbsoluteError
from sklearn.metrics import precision_score, accuracy_score, f1_score
from sklearn.dummy import DummyClassifier
import random
import torch as th

class Evaluator:
    def __init__(
            self,
            tasks,
            metrics,
            device,
            conf=None):
        self.tasks = tasks
        self.metrics = metrics
        self.device = device
        self.conf = conf

    def calculate_metrics(self, preds, labs):
        results = {}
        for task, metric in zip(self.tasks, self.metrics):
            if task == 'los_real':
                if metric == 'mae':
                    results['los_mae'] = self.mean_absolute_error(preds, labs)
            if task == 'los_binary':
                if metric == 'acc':
                    results['los_acc'] = self.binary_acc(preds['los_binary'], labs['los_binary'])
                if metric == 'f1':
                    results['los_f1'] = self.binary_f1(preds['los_binary'], labs['los_binary'])
            if task == 'los_category':
                if metric == 'acc':
                    results['los_acc'] = self.multi_acc(th.argmax(preds['los_category'], dim=1), labs['los_category'].squeeze())

        return results

    def binary_acc(self, preds, labs):
        metric = BinaryAccuracy().to(self.device)
        return metric(preds, labs).item()

    def multi_acc(self, preds, labs):
        metric = MulticlassAccuracy(len(self.conf['classes']) + 1).to(self.device)
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

    def get_loader_labels(self, loader, task):
        if task in ['los_binary', 'los_category']:
            data_y = [targets[task].tolist() for _, _, _, _, targets, _ in loader]
            data_y = [item[0] for sublist in data_y for item in sublist]

        return data_y

    def get_dummy_preds(self, labels):
        dummy_clf = DummyClassifier(strategy="most_frequent")
        random_input = [random.randrange(1, 50, 1) for _ in range(len(labels))]
        clf = dummy_clf.fit(random_input, labels)
        preds = clf.predict(random_input)

        return preds

    def baseline_results(self, train, evalu, test):
        for task, metric in zip(self.tasks, self.metrics):
            if task == 'los_real':
                if metric == 'mae':
                    print(f'Baseline Mean MAE for Train: {round(self.get_max_mae(train), 3)}')
                    print(f'Baseline Mean MAE for Eval: {round(self.get_max_mae(evalu), 3)}')
                    print(f'Baseline Mean MAE for Test: {round(self.get_max_mae(test), 3)}')

            if task in ['los_binary', 'los_category']:
                train_y = self.get_loader_labels(train, task)
                evalu_y = self.get_loader_labels(evalu, task)
                test_y = self.get_loader_labels(test, task)

                train_preds = self.get_dummy_preds(train_y)
                evalu_preds = self.get_dummy_preds(evalu_y)
                test_preds = self.get_dummy_preds(test_y)

                if metric == 'acc':
                    print(f'Baseline Prec for Train: {round(accuracy_score(train_y, train_preds), 2)}')
                    print(f'Baseline Prec for Eval: {round(accuracy_score(evalu_y, evalu_preds), 2)}')
                    print(f'Baseline Prec for Test: {round(accuracy_score(test_y, test_preds), 2)}')
                if metric == 'f1':
                    print(f'Baseline F1 for Train: {round(f1_score(train_y, train_preds), 2)}')
                    print(f'Baseline F1 for Eval: {round(f1_score(evalu_y, evalu_preds), 2)}')
                    print(f'Baseline F1 for Test: {round(f1_score(test_y, test_preds), 2)}')
