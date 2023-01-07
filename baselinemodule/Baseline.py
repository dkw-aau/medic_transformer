from baselinemodule.BaselineDataset import BaselineDataset
from Utils.utils import load_corpus
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, accuracy_score, f1_score
from sklearn.metrics import confusion_matrix


class Baseline:
    def __init__(
            self,
            args):

        self.args = args
        self.task = args.task
        self.cls = args.cls

        # Load Corpus
        corpus = load_corpus(os.path.join(args.path['data_fold'], args.corpus_name))

        # Extraxt x hours of data
        corpus = corpus.get_subset_by_min_hours(min_hours=self.args.hours)
        corpus.cut_sequences_by_hours(hours=self.args.hours)
        corpus.substract_los_hours(hours=self.args.hours)

        metrics = ['acc']
        conf = {
            'los_binary_threshold': 5,
            'classes': [1, 2, 3, 4, 5, 6, 7, 14]
        }

        dataset = BaselineDataset(corpus, args, conf)
        self.train_x = dataset.train_x
        self.train_y = dataset.train_y
        self.test_x = dataset.test_x
        self.test_y = dataset.test_y

    def train(self):
        if self.cls == 'rfc':
            clf = RandomForestClassifier(max_depth=2, random_state=0)
        else:
            exit(f'Classifier: {self.cls} not implemented')

        print('Training Classifier')
        clf.fit(self.train_x, self.train_y)

        test_preds = clf.predict(self.test_x)

        # Print metric

        print(accuracy_score(self.test_y, test_preds))

        if self.task in ['los_binary', 'los_category']:
            print(confusion_matrix(self.test_y, test_preds))


