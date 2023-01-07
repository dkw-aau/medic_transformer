import copy
import os
import datetime
from bisect import bisect
from sklearn.model_selection import train_test_split

import pandas as pd

from Utils.utils import pickle_load


class Corpus:
    def __init__(self, data_path, sequences=None):
        self.data_path = data_path
        self.corpus_df = None
        self.vocabulary = None
        self.time_format = '%Y-%m-%d %H:%M:%S'

        if sequences:
            self.sequences = sequences
        else:
            self.sequences = None

    def create_vocabulary(self):
        tokens = set()
        for seq in self.sequences:
            tokens.update(seq.event_tokens)

        special_tokens = {
            0: 'CLS',
            1: 'UNK',
            2: 'SEP',
            3: 'PAD',
            4: 'MASK'
        }

        tokens.difference_update(set(special_tokens.values()))

        index2token = special_tokens
        token2index = {val: key for key, val in special_tokens.items()}
        for index, token in enumerate(tokens, len(special_tokens)):
            index2token[index] = token
            token2index[token] = index

        self.vocabulary = {
            'index2token': index2token,
            'token2index': token2index
        }

    def get_vocabulary(self):
        self.create_vocabulary()

        return self.vocabulary

    def load_sequences(self, file_name):
        self.sequences = pickle_load(os.path.join(self.data_path, f'{file_name}.pkl'))

    def unpack_corpus_df(self):
        for seq, corp_df in zip(self.sequences, self.corpus_df.itertuples()):
            seq.event_tokens = corp_df.token.split(';')
            seq.event_types = corp_df.event_type.split(';')
            seq.event_pos_ids = [int(x) for x in corp_df.event_pos.split(';')]
            seq.event_tokens_orig = corp_df.token_orig.split(';')
            seq.event_values = [float(item) for item in corp_df.event_value.split(';')]
            seq.event_times = [datetime.datetime.strptime(event, self.time_format) for event in corp_df.event_time.split(';')]

    def make_corpus_df(self):
        sequence_list = []
        for seq in self.sequences:
            event_token_string = ';'.join(seq.event_tokens)
            event_token_orig_string = ';'.join(seq.event_tokens_orig)
            event_times = [event_time.strftime(self.time_format) for event_time in seq.event_times]
            event_time_string = ';'.join(event_times)
            event_type_string = ';'.join(seq.event_types)
            event_value_string = ';'.join([str(item) for item in seq.event_values])
            event_pos_string = ';'.join([str(x) for x in seq.event_pos_ids])
            sequence_list.append({'token': event_token_string, 'token_orig': event_token_orig_string, 'event_time': event_time_string, 'event_type': event_type_string, 'event_value': event_value_string, 'event_pos': event_pos_string})
            seq.event_tokens = None
            seq.event_times = None
            seq.event_types = None
            seq.event_tokens_orig = None
            seq.event_values = None
            seq.event_pos_ids = None

        self.corpus_df = pd.DataFrame.from_dict(sequence_list)

    def count_requires_hosp(self):
        return sum([1 if seq.req_hosp else 0 for seq in self.sequences])

    def get_sequence_lengths(self):
        return [len(seq.event_tokens) for seq in self.sequences]

    def get_num_sequences(self):
        return len(self.sequences)

    def cut_sequences_by_hours(self, hours):
        for seq in self.sequences:
            seq.cut_by_hours(hours)

    def substract_los_hours(self, hours):
        for seq in self.sequences:
            seq.length_of_stay = (seq.length_of_stay * 24 - hours) / 24

    def cut_sequences_by_apriori(self):
        for seq in self.sequences:
            seq.event_times = seq.event_times[0:seq.apriori_len]
            seq.event_tokens = seq.event_tokens[0:seq.apriori_len]

    def get_subset_by_uns_diag(self):
        new_corpus = copy.deepcopy(self)
        sequences = self.sequences.copy()
        new_sequences = []

        for seq in sequences:
            if seq.uns_diag:
                new_sequences.append(seq)

        new_corpus.sequences = new_sequences

        return new_corpus

    def get_subset_by_req_hosp(self):
        new_corpus = copy.deepcopy(self)
        sequences = self.sequences.copy()
        new_sequences = []

        for seq in sequences:
            if seq.req_hosp:
                new_sequences.append(seq)

        new_corpus.sequences = new_sequences

        return new_corpus

    def get_subset_by_mortality_30(self):
        new_corpus = copy.deepcopy(self)
        sequences = self.sequences.copy()
        new_sequences = []

        for seq in sequences:
            if seq.mortality_30:
                new_sequences.append(seq)

        new_corpus.sequences = new_sequences

        return new_corpus

    def get_subset_by_min_hours(self, min_hours):
        new_corpus = copy.deepcopy(self)
        sequences = self.sequences.copy()
        new_sequences = []

        for seq in sequences:
            if seq.length_of_stay * 24 >= min_hours:
                new_sequences.append(seq)

        new_corpus.sequences = new_sequences

        return new_corpus

    def split_train_test(self, train_size=0.8):
        train, test = train_test_split(self.sequences, test_size=1 - train_size, random_state=42)
        return train, test

    def split_train_eval_test(self, train_size):
        train, eval = train_test_split(self.sequences, test_size=1 - train_size, random_state=42)
        eval, test = train_test_split(eval, test_size=0.5, random_state=42)

        return train, eval, test
