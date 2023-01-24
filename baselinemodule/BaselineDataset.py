import math
import time
from bisect import bisect

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import chi2, f_regression
from sklearn.feature_selection import SelectKBest


class BaselineDataset:
    def __init__(
            self,
            corpus,
            args,
            conf):

        self.strategy = args.strategy
        self.imputation = args.imputation
        self.imputation = args.imputation
        self.corpus = corpus
        self.scaler = args.scaler
        self.feature_select = args.feature_select
        self.conf = conf
        self.tokens = None
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None

        # Get distinct token types
        print("Getting distinct tokens")
        self.get_distinct_orig_tokens()

        print('Grouping sequences by tokens')
        # Group values by token types
        self.group_sequence_values()

        # Extract samples from sequences
        print('Extracting samples from sequences')
        self.extract_samples()

        # Split samples
        print('Splitting data into train and test')
        self.train_x, _, self.test_x = self.corpus.split_train_eval_test()

        print('Splitting samples into data and labels')
        self.train_x, self.train_y, self.test_x, self.test_y = self.split_data()

        # Do data imputation
        print('Imputing missing values')
        self.train_x, self.test_x = self.impute_samples()

        # Scale data
        print('Scaling the data')
        self.train_x, self.test_x = self.scale()

        print('Feature selection')
        self.train_x, self.test_x = self.feature_reduct()

    def split_data(self):
        train_x = []
        train_y = []
        for seq in self.train_x:
            train_x.append(seq.data_x)
            train_y.append(seq.label)

        test_x = []
        test_y = []
        for seq in self.test_x:
            test_x.append(seq.data_x)
            test_y.append(seq.label)

        return train_x, train_y, test_x, test_y

    def feature_reduct(self):
        if self.feature_select == 'chi2':
            chi2_selector = SelectKBest(chi2, k=50)
            chi2_selector.fit(self.train_x, self.train_y)
            train_x = chi2_selector.transform(self.train_x)
            test_x = chi2_selector.transform(self.test_x)
        elif self.feature_select == 'f_reg':
            fs = SelectKBest(score_func=f_regression, k=50)
            # learn relationship from training data
            fs.fit(self.train_x, self.train_y)
            # transform train input data
            train_x = fs.transform(self.train_x)
            # transform test input data
            test_x = fs.transform(self.test_x)
        else:
            train_x = self.train_x
            test_x = self.test_x

        return train_x, test_x

    def scale(self):
        if self.scaler == 'standard':
            scaler = StandardScaler()
        elif self.scaler == 'min-max':
            scaler = MinMaxScaler()
        elif self.scaler == 'none':
            pass
        else:
            exit(f'Scaler {self.scaler} not implemented')

        if self.scaler != 'none':
            scaler.fit(self.train_x)
            train = scaler.transform(self.train_x)
            test = scaler.transform(self.test_x)

        return train, test

    def impute_samples(self):
        if self.imputation in ['mean', 'median']:
            imp = SimpleImputer(missing_values=np.nan, strategy=self.imputation)
        elif self.imputation == 'none':
            pass
        else:
            exit(f'Impute strategy {self.imputation} not implemented')

        if self.imputation != 'none':
            imp.fit(self.train_x)
            train = imp.transform(self.train_x)
            test = imp.transform(self.test_x)

        return train, test

    def extract_samples(self):
        for seq in self.corpus.sequences:
            sample = []
            for token in self.tokens:
                if self.strategy == 'last':
                    if token in seq.token_value_dict:
                        sample.append(seq.token_value_dict[token])
                    else:
                        sample.append(np.nan)
                elif self.strategy == 'min-max':
                    if token in seq.token_value_dict:
                        sample.append(max(seq.token_value_dict[token]))
                        sample.append(min(seq.token_value_dict[token]))
                    else:
                        sample.append(np.nan)
                        sample.append(np.nan)
                elif self.strategy == 'min':
                    if token in seq.token_value_dict:
                        sample.append(min(seq.token_value_dict[token]))
                    else:
                        sample.append(np.nan)
                elif self.strategy == 'max':
                    if token in seq.token_value_dict:
                        sample.append(max(seq.token_value_dict[token]))
                    else:
                        sample.append(np.nan)
                elif self.strategy == 'avg':
                    if token in seq.token_value_dict:
                        sample.append(sum(seq.token_value_dict[token]) / len(seq.token_value_dict[token]))
                    else:
                        sample.append(np.nan)
                else:
                    exit('Strategy not implemented')

            seq.data_x = sample

    def get_distinct_orig_tokens(self):
        tokens = set()
        for seq in self.corpus.sequences:
            tokens.update(set(seq.event_tokens_orig))

        # Remove some tokens from the set
        tokens_to_remove = ['CLS', 'SEP', 'MASK', 'UNK', 'PAD']
        tokens.difference_update(tokens_to_remove)

        # Add age and gender
        tokens.update({'age', 'gender'})
        self.tokens = tokens

    def group_sequence_values(self):
        for seq in self.corpus.sequences:
            seq.group_token_values()

