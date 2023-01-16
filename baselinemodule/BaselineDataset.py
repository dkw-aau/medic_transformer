import math
import time
from bisect import bisect

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import chi2
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

        print('Setting up label task')
        self.data_y = self.setup_label_task()

        # Split samples
        print('Splitting data into train and test')
        self.train_x, self.test_x, self.train_y, self.test_y = self.split_train_test(train_size=0.8)

        # Do data imputation
        print('Imputing missing values')
        self.train_x, self.test_x = self.impute_samples()

        # Scale data
        print('Scaling the data')
        self.train_x, self.test_x = self.scale()

        print('Feature selection')
        self.train_x, self.test_x = self.feature_reduct()

    def feature_reduct(self):
        if self.feature_select == 'chi2':
            chi2_selector = SelectKBest(chi2, k=50)
            chi2_selector.fit(self.train_x, self.train_y)
            train_x = chi2_selector.transform(self.train_x)
            test_x = chi2_selector.transform(self.test_x)
        else:
            train_x = self.train_x
            test_x = self.test_x

        return train_x, test_x

    def setup_label_task(self):
        data_y = []
        for seq in self.sequences:
            data_y.append(seq.label)

        return data_y

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

        # Add age and gender
        tokens.update({'age', 'gender'})
        self.tokens = tokens

    def group_sequence_values(self):
        for seq in self.sequences:
            seq.group_token_values(self.tokens)

    def split_train_test(self, train_size=0.8):
        train_x, test_x, train_y, test_y = train_test_split(self.data_x, self.data_y, stratify=self.data_y, test_size=1-train_size, random_state=42)
        return train_x, test_x, train_y, test_y

    def split_train_eval_test(self, train_size):
        train, eval = train_test_split(self.data_x, self.data_y, stratify=self.data_y, test_size=1 - train_size, random_state=42)
        eval, test = train_test_split(eval, test_size=0.5, random_state=42)

        return train, eval, test
