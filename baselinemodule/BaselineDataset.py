import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import chi2, f_regression
from sklearn.feature_selection import SelectKBest


class BaselineDataset:
    def __init__(self, corpus, corpus_conf):

        self.corpus = corpus
        self.corpus_conf = corpus_conf
        self.tokens = None
        self.train_x, self.test_x = None, None
        self.train_y_bin, self.train_y_cat, self.train_y_real = None, None, None
        self.test_y_bin, self.test_y_cat, self.test_y_real = None, None, None
        self.train_x_bin, self.train_x_cat, self.train_x_real = None, None, None
        self.test_x_bin, self.test_x_cat, self.test_x_real = None, None, None

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
        train_x, train_y_bin, train_y_cat, train_y_real, test_x, test_y_bin, test_y_cat, test_y_real = self.split_data()
        self.train_x, self.test_x = train_x, test_x
        self.train_y_bin, self.train_y_cat, self.train_y_real = train_y_bin, train_y_cat, train_y_real
        self.test_y_bin, self.test_y_cat, self.test_y_real = test_y_bin, test_y_cat, test_y_real

        # Do data imputation
        print('Imputing missing values')
        self.train_x, self.test_x = self.impute_samples()

        # Scale data
        print('Scaling the data')
        self.train_x, self.test_x = self.scale()

        print('Feature selection')
        train_x_bin, train_x_cat, train_x_real, test_x_bin, test_x_cat, test_x_real = self.feature_reduct()
        self.train_x_bin, self.train_x_cat, self.train_x_real = train_x_bin, train_x_cat, train_x_real
        self.test_x_bin, self.test_x_cat, self.test_x_real = test_x_bin, test_x_cat, test_x_real

    def split_data(self):
        train_x = []
        train_y_bin = []
        train_y_cat = []
        train_y_real = []
        for seq in self.train_x:
            train_x.append(seq.data_x)
            self.corpus_conf.update({'task': 'binary'})
            seq.create_label(self.corpus_conf)
            train_y_bin.append(seq.label)
            self.corpus_conf.update({'task': 'category'})
            seq.create_label(self.corpus_conf)
            train_y_cat.append(seq.label)
            self.corpus_conf.update({'task': 'real'})
            seq.create_label(self.corpus_conf)
            train_y_real.append(seq.label)

        test_x = []
        test_y_bin = []
        test_y_cat = []
        test_y_real = []
        for seq in self.test_x:
            test_x.append(seq.data_x)
            self.corpus_conf.update({'task': 'binary'})
            seq.create_label(self.corpus_conf)
            test_y_bin.append(seq.label)
            self.corpus_conf.update({'task': 'category'})
            seq.create_label(self.corpus_conf)
            test_y_cat.append(seq.label)
            self.corpus_conf.update({'task': 'real'})
            seq.create_label(self.corpus_conf)
            test_y_real.append(seq.label)

        return train_x, train_y_bin, train_y_cat, train_y_real, test_x, test_y_bin, test_y_cat, test_y_real

    def feature_reduct(self):
        # Binary and categorical data selections
        chi2_selector = SelectKBest(chi2, k=50)
        chi2_selector.fit(self.train_x, self.train_y_bin)
        train_x_bin = chi2_selector.transform(self.train_x)
        test_x_bin = chi2_selector.transform(self.test_x)

        # Binary and categorical data selections
        chi2_selector = SelectKBest(chi2, k=50)
        chi2_selector.fit(self.train_x, self.train_y_cat)
        train_x_cat = chi2_selector.transform(self.train_x)
        test_x_cat = chi2_selector.transform(self.test_x)

        # Real prediction data selection
        fs = SelectKBest(score_func=f_regression, k=50)
        fs.fit(self.train_x, self.train_y_real)
        train_x_real = fs.transform(self.train_x)
        test_x_real = fs.transform(self.test_x)

        return train_x_bin, train_x_cat, train_x_real, test_x_bin, test_x_cat, test_x_real

    def scale(self):
        scaler = MinMaxScaler()

        scaler.fit(self.train_x)
        train = scaler.transform(self.train_x)
        test = scaler.transform(self.test_x)

        return train, test

    def impute_samples(self):
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp.fit(self.train_x)
        train = imp.transform(self.train_x)
        test = imp.transform(self.test_x)

        return train, test

    def extract_samples(self):
        for seq in self.corpus.sequences:
            sample = []
            for token in self.tokens:
                if token in seq.token_value_dict:
                    sample.append(seq.token_value_dict[token])
                else:
                    sample.append(np.nan)

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

