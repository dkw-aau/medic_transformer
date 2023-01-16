from baselinemodule.BaselineDataset import BaselineDataset
from Utils.utils import load_corpus, save_baseline_data, load_baseline_date
from sklearn.neural_network import MLPClassifier
import os

from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.svm import SVC


class Baseline:
    def __init__(
            self,
            args):

        self.args = args
        self.cls = args.cls

        self.conf = {
            'task': 'category',
            'binary_thresh': 2,
            'cats': [2, 7],
            'years': [2018, 2019, 2020, 2021],
            'types': ['apriori', 'adm', 'proc', 'vital', 'lab'],  # 'apriori', 'vital', 'diag', 'apriori', 'adm', 'proc', 'lab'
            'max_hours': 24
        }

        # If use_saved, load previously processed data
        if self.args.use_saved:
            print('Using pre-loaded dataset')
            self.train_x, self.train_y, self.test_x, self.test_y = load_baseline_date(self.args.path['out_fold'], self.conf['task'])
            return

        # Load Corpus
        print('Loading Corpus')
        corpus = load_corpus(os.path.join(args.path['data_fold'], args.corpus_name))

        # Prepare Corpus
        print('Preparing Corpus')
        _ = corpus.prepare_corpus(
            self.conf
        )
        corpus.create_pos_ids(event_dist=300)
        corpus.create_train_evel_test_idx(train_size=0.8)

        print('Creating dataset from sequences')
        dataset = BaselineDataset(corpus, args, self.conf)
        self.train_x = dataset.train_x
        self.train_y = dataset.train_y
        self.test_x = dataset.test_x
        self.test_y = dataset.test_y

        # Save baseline data to file
        save_baseline_data(self.train_x, self.train_y, self.test_x, self.test_y, self.conf['task'], self.args.path['out_fold'])

    def train(self):
        if self.cls == 'rfc':
            clf = RandomForestClassifier(random_state=0)
        elif self.cls == 'nn':
            clf = MLPClassifier()
        elif self.cls == 'svc':
            clf = SVC(probability=True)
        else:
            exit(f'Classifier: {self.cls} not implemented')

        print('\nTraining Classifier')
        clf.fit(self.train_x, self.train_y)

        # Print metric
        if self.conf['task'] in ['binary', 'category']:
            if self.conf['task'] == 'binary':
                probs = clf.predict_proba(self.test_x)[:, 1]
            else:
                probs = clf.predict_proba(self.test_x)
            score = roc_auc_score(self.test_y, probs, multi_class='ovo')
            print(f'ROC-AUC: {score}')

            preds = clf.predict(self.test_x)
            score = accuracy_score(self.test_y, preds)
            print(f'Acc: {score}')

            if self.conf['task'] == 'binary':
                score = f1_score(self.test_y, preds)
            else:
                score = f1_score(self.test_y, preds, average='macro')
            print(f'F1 Score: {score}')

        elif self.conf['task'] in ['real']:
            pass

        print('Saving fitted model')
        dump(clf, os.path.join(self.args.path['out_fold'], f'{self.cls}_{self.conf["task"]}.joblib'))
