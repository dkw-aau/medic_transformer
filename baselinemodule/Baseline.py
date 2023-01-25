from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from baselinemodule.BaselineDataset import BaselineDataset
from sklearn.svm import SVC, SVR
from Utils.Corpus import Corpus
from joblib import dump
import os


class Baseline:
    def __init__(self, args):
        self.args = args

        # Corpus configuration
        self.corpus_conf = {
            'task': 'binary',
            'binary_thresh': args.binary_thresh,
            'cats': args.categories,
            'years': args.years,
            'types': args.types,
            'seq_hours': args.seq_hours,
            'clip_los': args.clip_los
        }
        # Prepare corpus
        print('Loading Corpus')
        self.corpus = Corpus(
            data_path=args.path['data_fold'],
            file_names=args.file_names
        )
        self.vocab = self.corpus.prepare_corpus(self.corpus_conf)

        print('Creating dataset from sequences')
        self.dataset = BaselineDataset(self.corpus, self.corpus_conf)

    def train(self):
        # For each task
        for task in ['binary', 'category', 'real']:
            print('\n------------------------')
            print(f'Training for task: {task}')
            print('------------------------\n')

            # Setup train and eval data
            train_x, train_y, test_x, test_y, clf = [None] * 5
            if task == 'binary':
                train_x, train_y = self.dataset.train_x_bin, self.dataset.train_y_bin
                test_x, test_y = self.dataset.test_x_bin, self.dataset.test_y_bin
            if task == 'category':
                train_x, train_y = self.dataset.train_x_cat, self.dataset.train_y_cat
                test_x, test_y = self.dataset.test_x_cat, self.dataset.test_y_cat
            if task == 'real':
                train_x, train_y = self.dataset.train_x_real, self.dataset.train_y_real
                test_x, test_y = self.dataset.test_x_real, self.dataset.test_y_real

            # For each classifier
            for cls in ['rfc', 'ann', 'svm']:

                # Setup classifier
                if task in ['binary', 'category']:
                    if cls == 'rfc':
                        clf = RandomForestClassifier(random_state=42)
                    elif cls == 'ann':
                        clf = MLPClassifier(random_state=42)
                    elif cls == 'svm':
                        clf = SVC(probability=True)

                elif task == 'real':
                    if cls == 'rfc':
                        clf = RandomForestRegressor(random_state=42)
                    elif cls == 'ann':
                        clf = MLPRegressor(random_state=42)
                    elif cls == 'svm':
                        clf = SVR()

                print(f'\nTraining Classifier: {cls}')
                clf.fit(train_x, train_y)

                # Print metrics
                print('Calculating Metrics')
                if task in ['binary', 'category']:
                    probs = clf.predict_proba(test_x)[:, 1] if task == 'binary' else clf.predict_proba(test_x)
                    score = roc_auc_score(test_y, probs, multi_class='ovo')
                    print(f'{cls} - ROC-AUC: {score}')

                    preds = clf.predict(test_x)
                    score = f1_score(test_y, preds) if task == 'binary' else f1_score(test_y, preds, average='macro')
                    print(f'{cls} - F1 Score: {score}')

                elif task in ['real']:
                    preds = clf.predict(test_x)
                    mse_score = mean_squared_error(test_y, preds)
                    mae_score = mean_absolute_error(test_y, preds)
                    print(f'{cls} - MSE: {mse_score}')
                    print(f'{cls} - MAE: {mae_score}')

                print('Saving fitted model')
                dump(clf, os.path.join(self.args.path['out_fold'], f'{cls}_{task}.joblib'))
