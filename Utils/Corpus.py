import copy
import math
import os
import datetime
import random
import time

import numpy as np
import pandas as pd
from Utils.utils import pickle_load
from sklearn.model_selection import train_test_split


class Corpus:
    def __init__(self, data_path):
        self.data_path = data_path
        self.corpus_df = None
        self.vocabulary = None
        self.time_format = '%Y-%m-%d %H:%M:%S'
        self.train_idx = None
        self.eval_idx = None
        self.test_idx = None
        self.sequences = None
        self.scaler = None

    def create_vocabulary(self):
        tokens = set()
        for seq in self.sequences:
            tokens.update(seq.event_tokens)

        special_tokens = {
            'CLS',
            'UNK',
            'SEP',
            'PAD',
            'MASK'
        }

        all_tokens = sorted(list(tokens.union(special_tokens)))

        index2token = {}
        token2index = {}
        for index, token in enumerate(all_tokens):
            index2token[index] = token
            token2index[token] = index

        self.vocabulary = {
            'index2token': index2token,
            'token2index': token2index
        }

    def get_vocabulary(self):
        self.create_vocabulary()
        print(f'Length of Vocabulary: {len(self.vocabulary["index2token"])}')
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

    def get_subset_by_mortality_30(self):
        new_corpus = copy.deepcopy(self)
        sequences = self.sequences.copy()
        new_sequences = []

        for seq in sequences:
            if seq.mortality_30:
                new_sequences.append(seq)

        new_corpus.sequences = new_sequences

        return new_corpus

    def get_labels(self):
        labels = []
        for seq in self.sequences:
            labels.append(seq.label)

        return labels

    def create_train_evel_test_idx(self, train_size=0.8, task=None):

        # Sample indexes
        indexes = list(range(0, len(self.sequences)))

        # Shuffle examples
        random.shuffle(indexes)
        sequences = list(np.array(self.sequences)[indexes])

        stratify = True
        labels = [seq.label for seq in sequences]
        if task == 'real':
            # Make stratification on bins of 5 days
            labels = [int(math.floor(los / 5)) for los in labels]
            min_group = min([labels.count(num) for num in set(labels)])
            stratify = True if min_group >= 10 else False

        # Split data
        if stratify:
            train_idx, eval_idx, _, eval_labs = train_test_split(indexes, labels, stratify=labels, test_size=1-train_size, random_state=40)
            eval_idx, test_idx = train_test_split(eval_idx, stratify=eval_labs, test_size=0.5, random_state=40)
        else:
            train_idx, eval_idx = train_test_split(indexes, test_size=1 - train_size, random_state=40)
            eval_idx, test_idx = train_test_split(eval_idx, test_size=0.5, random_state=40)

        self.train_idx = train_idx
        self.eval_idx = eval_idx
        self.test_idx = test_idx

    def split_train_eval_test(self):
        train = list(np.array(self.sequences)[self.train_idx])
        evalu = list(np.array(self.sequences)[self.eval_idx])
        test = list(np.array(self.sequences)[self.test_idx])
        print(f'Length of data split {len(train)}/{len(evalu)}/{len(test)}')
        print(f'Train Mean LOS: {sum([seq.length_of_stay for seq in train]) / len(train)}')
        print(f'Eval Mean LOS: {sum([seq.length_of_stay for seq in evalu]) / len(evalu)}')
        print(f'Test Mean LOS: {sum([seq.length_of_stay for seq in test]) / len(test)}')
        return train, evalu, test

    def randomize_sequences(self):
        for sequence in self.sequences:
            random.shuffle(sequence.event_tokens)

    def create_pos_ids(self, event_dist=60):
        for seq in self.sequences:
            seq.create_position_ids(event_dist)

    def prepare_corpus(self, conf=None):
        print('Preparing Corpus')
        whole_vocabulary = self.get_vocabulary()

        new_sequences = []
        # Process sequences
        for seq in self.sequences:

            # Remove Unwanted years
            if seq.ed_start.year not in conf['years']:
                continue

            # Skip if los is shorter than max_hours
            if conf['seq_hours'] and seq.length_of_stay * 24 <= conf['seq_hours']:
                continue

            # Skip if los is an outlier
            #if seq.length_of_stay > 100:
            #    continue

            # Remove if sequence longer than max_hours
            seq.cut_by_hours(conf['seq_hours'])

            # Remove unwanted types
            del_indexes = [i for i, t in enumerate(seq.event_types) if t not in conf['types']]
            seq.remove_indexes(del_indexes)

            # Change LOS based on max_hours
            if conf['seq_hours']:
                seq.minus_los(conf['seq_hours'])

            # Clip LOS
            if conf['clip_los'] != -1:
                seq.clip_los(conf['clip_los'])

            # Create label
            seq.create_label(conf)

            new_sequences.append(seq)

        self.sequences = new_sequences

        # Create event positions
        self.create_pos_ids(event_dist=300)

        # Split data into train eval and test
        self.create_train_evel_test_idx(train_size=0.8, task=conf['task'])
        return whole_vocabulary
