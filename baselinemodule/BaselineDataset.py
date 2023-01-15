import math
import time
from bisect import bisect

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import chi2


class BaselineDataset:
    def __init__(
            self,
            corpus,
            args,
            conf):

        self.strategy = args.strategy
        self.imputation = args.imputation
        self.imputation = args.imputation
        self.sequences = corpus.sequences
        self.scaler = args.scaler
        self.feature_select = args.feature_select
        self.conf = conf
        self.tokens = None
        self.data_x = []
        self.data_y = []

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
        train_x, test_x, train_y, test_y = self.split_train_test(train_size=0.8)

        # Do data imputation
        print('Imputing missing values')
        train_x, test_x = self.impute_samples(train_x, test_x)

        # Scale data
        print('Scaling the data')
        self.train_x, self.test_x = self.scale(train_x, test_x)

        print('Setting up label task')
        self.train_y, self.test_y = self.setup_label_task(train_y, test_y)

        print('Feature selection')
        self.feature_reduct()

    def feature_reduct(self):
        if self.feature_select == 'chi2':
            scores, p_values = chi2(self.train_x, self.train_y)
            print('Hello')
        else:
            pass

    def setup_label_task(self, train_y, test_y):
        if self.conf['task'] == 'real':
            pass
        elif self.conf['task'] == 'binary':
            train_y = [1 if los > self.conf['binary_thresh'] else 0 for los in train_y]
            test_y = [1 if los > self.conf['binary_thresh'] else 0 for los in test_y]
        elif self.conf['task'] == 'category':
            train_y = [bisect(self.conf['cats'], los) for los in train_y]
            test_y = [bisect(self.conf['cats'], los) for los in test_y]
        else:
            exit(f'Task: {self.conf["task"]} not implemented')

        return train_y, test_y

    def scale(self, train, test):
        if self.scaler == 'standard':
            scaler = StandardScaler()
        elif self.scaler == 'min-max':
            scaler = MinMaxScaler()
        elif self.scaler == 'none':
            pass
        else:
            exit(f'Scaler {self.scaler} not implemented')

        if self.scaler != 'none':
            scaler.fit(train)
            train = scaler.transform(train)
            test = scaler.transform(test)

        return train, test

    def impute_samples(self, train, test):
        if self.imputation in ['mean', 'median']:
            imp = SimpleImputer(missing_values=np.nan, strategy=self.imputation)
        elif self.imputation == 'none':
            pass
        else:
            exit(f'Impute strategy {self.imputation} not implemented')

        if self.imputation != 'none':
            imp.fit(train)
            train = imp.transform(train)
            test = imp.transform(test)

        return train, test

    def extract_samples(self):
        for seq in self.sequences:
            sample = []
            for token in self.tokens:
                if self.strategy == 'min-max':
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
            self.data_x.append(sample)
            self.data_y.append(seq.length_of_stay)

    def get_distinct_orig_tokens(self):
        tokens = set()
        for seq in self.sequences:
            tokens.update(set(seq.event_tokens_orig))

        # Remove some tokens from the set
        tokens_to_remove = ['CLS', 'SEP', 'MASK', 'UNK', 'PAD']
        tokens.difference_update(tokens_to_remove)
        self.tokens = tokens

    def group_sequence_values(self):
        for seq in self.sequences:
            seq.group_token_values(self.tokens)

    def split_train_test(self, train_size=0.8):
        train_x, test_x, train_y, test_y = train_test_split(self.data_x, self.data_y, test_size=1 - train_size, random_state=42)
        return train_x, test_x, train_y, test_y

    def split_train_eval_test(self, train_size):
        train, eval = train_test_split(self.data_x, self.data_y, test_size=1 - train_size, random_state=42)
        eval, test = train_test_split(eval, test_size=0.5, random_state=42)

        return train, eval, test
