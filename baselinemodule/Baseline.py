from baselinemodule.BaselineDataset import BaselineDataset
from Utils.utils import load_corpus, save_baseline_data, load_baseline_date
from sklearn.neural_network import MLPClassifier
import os
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


class Baseline:
    def __init__(
            self,
            args):

        self.args = args
        self.task = args.task
        self.cls = args.cls

        # If use_saved, load previously processed data
        if self.args.use_saved:
            self.train_x, self.train_y, self.test_x, self.test_y = load_baseline_date(self.args.path['out_fold'])
            return

        # Load Corpus
        print('Loading Corpus')
        corpus = load_corpus(os.path.join(args.path['data_fold'], args.corpus_name))
        corpus.create_train_evel_test_idx(train_size=0.8)

        # Extract x hours of data
        print(f'Extracting {self.args.hours} hours data from each subject')
        corpus = corpus.get_subset_corpus(min_hours=args.hours)
        #corpus.cut_sequences_by_apriori()

        conf = {
            'los_binary_threshold': 5,
            'classes': [1, 2, 3, 4, 5, 6, 7, 14]
        }

        print('Creating dataset from sequences')
        dataset = BaselineDataset(corpus, args, conf)
        self.train_x = dataset.train_x
        self.train_y = dataset.train_y
        self.test_x = dataset.test_x
        self.test_y = dataset.test_y

        # Save baseline data to file
        save_baseline_data(self.train_x, self.train_y, self.test_x, self.test_y, self.args.path['out_fold'])

    def train(self):
        if self.cls == 'rfc':
            clf = RandomForestClassifier(random_state=0)
        elif self.cls == 'nn':
            clf = MLPClassifier()
        else:
            exit(f'Classifier: {self.cls} not implemented')

        print('\nTraining Classifier')
        clf.fit(self.train_x, self.train_y)

        test_preds = clf.predict(self.test_x)

        # Print metric

        print(accuracy_score(self.test_y, test_preds))

        if self.task in ['los_binary', 'los_category']:
            print(confusion_matrix(self.test_y, test_preds))


