import math
import os
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from Utils.Sequence import Sequence


class Corpus:
    def __init__(self, data_path, file_names):
        self.data_path = data_path
        self.file_names = file_names
        self.vocabulary = None
        self.time_format = '%Y-%m-%d %H:%M:%S'
        self.train_idx = None
        self.eval_idx = None
        self.test_idx = None
        self.sequences = []
        self.scaler = None

        self.create_sequences()

    def create_sequences(self):
        # Load data and hosp files
        if 'parquet' in self.file_names['patients']:
            patients = pd.read_parquet(os.path.join(self.data_path, self.file_names['patients']))
        else:
            patients = pd.read_csv(os.path.join(self.data_path, self.file_names['patients']))
        if 'parquet' in self.file_names['data']:
            data = pd.read_parquet(os.path.join(self.data_path, self.file_names['data']))
        else:
            data = pd.read_csv(os.path.join(self.data_path, self.file_names['data']))

        # Group data by patientid
        grouped_data = data.groupby(by='sequence_id')

        # Creating sequences
        for _, patient in patients.iterrows():
            seq_id = patient['sequence_id']
            if seq_id in grouped_data.groups.keys():
                patient_data = grouped_data.get_group(seq_id)

                self.sequences.append(
                    Sequence(
                        patient['age'],
                        patient['sex'],
                        patient['hosp_start'],
                        patient['los'],
                        patient_data['token'].to_numpy(),
                        patient_data['token_orig'].to_numpy(),
                        patient_data['event_type'].to_numpy(),
                        patient_data['event_value'].to_numpy(),
                        patient_data['event_time'].to_numpy(),
                    )
                )

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

    def get_labels(self):
        labels = []
        for seq in self.sequences:
            labels.append(seq.label)

        return labels

    def create_train_eval_test_idx(self, train_size=0.8):

        # Sample indexes
        indexes = list(range(0, len(self.sequences)))

        # Shuffle examples
        random.shuffle(indexes)
        sequences = list(np.array(self.sequences)[indexes])

        labels = [seq.los for seq in sequences]

        # Make stratification on bins of 2 days if enough
        labels = [int(math.floor(los / 2)) for los in labels]
        min_group = min([labels.count(num) for num in set(labels)])
        stratify = True if min_group >= 10 else False

        # Split data
        if stratify:
            train_idx, eval_idx, _, eval_labs = train_test_split(indexes, labels, stratify=labels, test_size=1-train_size, random_state=42)
            eval_idx, test_idx = train_test_split(eval_idx, stratify=eval_labs, test_size=0.5, random_state=42)
        else:
            train_idx, eval_idx = train_test_split(indexes, test_size=1 - train_size, random_state=42)
            eval_idx, test_idx = train_test_split(eval_idx, test_size=0.5, random_state=42)

        self.train_idx = train_idx
        self.eval_idx = eval_idx
        self.test_idx = test_idx

    def split_train_eval_test(self):
        train = list(np.array(self.sequences)[self.train_idx])
        evalu = list(np.array(self.sequences)[self.eval_idx])
        test = list(np.array(self.sequences)[self.test_idx])
        print(f'Length of data split {len(train)}/{len(evalu)}/{len(test)}')
        print(f'Train Mean LOS: {sum([seq.los for seq in train]) / len(train)}')
        print(f'Eval Mean LOS: {sum([seq.los for seq in evalu]) / len(evalu)}')
        print(f'Test Mean LOS: {sum([seq.los for seq in test]) / len(test)}')
        return train, evalu, test

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
            if seq.hosp_start.year not in conf['years']:
                continue

            # Skip if los is shorter than max_hours
            if conf['seq_hours'] and seq.los * 24 <= conf['seq_hours']:
                continue

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
        self.create_train_eval_test_idx(train_size=0.8)
        return whole_vocabulary
