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
            2: 'PAD',
            3: 'MASK'
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

    def load_sequences(self, file_name):
        self.sequences = pickle_load(os.path.join(self.data_path, f'{file_name}.pkl'))

    def unpack_corpus_df(self):
        for seq, corp_df in zip(self.sequences, self.corpus_df.itertuples()):
            seq.event_tokens = corp_df.token.split(';')
            seq.event_times = [datetime.datetime.strptime(event, self.time_format) for event in corp_df.event_time.split(';')]

    def make_corpus_df(self):
        sequence_list = []
        for seq in self.sequences:
            event_token_string = ';'.join(seq.event_tokens)
            event_times = [event_time.strftime(self.time_format) for event_time in seq.event_times]
            event_time_string = ';'.join(event_times)
            sequence_list.append({'token': event_token_string, 'event_time': event_time_string})
            seq.event_tokens = None
            seq.event_times = None

        self.corpus_df = pd.DataFrame.from_dict(sequence_list)

    def count_requires_hosp(self):
        return sum([1 if seq.req_hosp else 0 for seq in self.sequences])

    def get_sequence_lengths(self):
        return [len(seq.event_tokens) for seq in self.sequences]

    def get_num_sequences(self):
        return len(self.sequences)

    def cut_sequences_by_hours(self, hours, minutes):
        for seq in self.sequences:
            end_time = seq.ed_start + datetime.timedelta(hours=hours, minutes=minutes)
            index = bisect(seq.event_times, end_time)
            seq.event_times = seq.event_times[0:index]
            seq.event_tokens = seq.event_tokens[0:index]

    def cut_sequences_by_apriori(self):
        for seq in self.sequences:
            seq.event_times = seq.event_times[0:seq.apriori_len]
            seq.event_tokens = seq.event_tokens[0:seq.apriori_len]

    def get_subset_corpus(self, req_hosp=None, uns_diag=None, mortality_30=None):
        subset_seq = self.sequences.copy()
        self.sequences = None
        new_corpus = copy.deepcopy(self)
        self.sequences = subset_seq

        if req_hosp is not None:
            subset_seq = [seq for seq in subset_seq if seq.req_hosp == req_hosp]
        if uns_diag is not None:
            subset_seq = [seq for seq in subset_seq if seq.uns_diag == uns_diag]
        if mortality_30 is not None:
            subset_seq = [seq for seq in subset_seq if seq.mortality_30 == mortality_30]

        new_corpus.sequences = subset_seq

        return new_corpus

    def get_data_split(self):
        train, eval = train_test_split(self.sequences, test_size=0.2, random_state=42)
        eval, test = train_test_split(eval, test_size=0.5, random_state=42)

        return train, eval, test



