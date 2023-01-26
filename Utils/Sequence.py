import datetime
from bisect import bisect

import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None


class Sequence:
    def __init__(self,
                 age,
                 sex,
                 hosp_start,
                 los,
                 event_tokens,
                 event_tokens_orig,
                 event_types,
                 event_values,
                 event_times):
        self.age = age
        self.sex = sex
        self.los = los
        self.hosp_start = hosp_start
        self.event_tokens = event_tokens
        self.event_tokens_orig = event_tokens_orig
        self.event_types = event_types
        self.event_values = event_values
        self.event_times = event_times

        self.event_pos_ids = None
        self.token_value_dict = None
        self.label = None

    def cut_by_hours(self, cutoff_hours=None):
        if cutoff_hours is not None and cutoff_hours >= 0:
            end_time = self.hosp_start + datetime.timedelta(hours=cutoff_hours, seconds=1)
            index = bisect(self.event_times, end_time)
            self.event_times = self.event_times[0:index]
            self.event_tokens_orig = self.event_tokens_orig[0:index]
            self.event_values = self.event_values[0:index]
            self.event_tokens = self.event_tokens[0:index]
            self.event_types = self.event_types[0:index]

    def minus_los(self, hours):
        self.los = (self.los * 24 - hours) / 24

    def group_token_values(self):
        token_value_dict = {}
        for token, value in zip(reversed(self.event_tokens_orig), reversed(self.event_values)):
            if token in token_value_dict:
                continue
            else:
                token_value_dict[token] = value

        # Add age and gender
        token_value_dict['age'] = self.age
        token_value_dict['gender'] = self.sex
        self.token_value_dict = token_value_dict

    def create_label(self, conf):
        lab = None
        if conf['task'] == 'binary':
            lab = 1 if self.los > conf['binary_thresh'] else 0
        elif conf['task'] == 'category':
            lab = bisect(conf['cats'], self.los)
        elif conf['task'] in ['real', 'mlm']:
            lab = self.los

        self.label = lab

    def create_position_ids(self, dist=60):
        tmp_id = 0
        tmp_time = self.hosp_start
        position_ids = []

        for event_time in self.event_times:
            if event_time < tmp_time + np.timedelta64(dist, 's'):
                position_ids.append(tmp_id)
            else:
                tmp_id += 1
                tmp_time = event_time
                position_ids.append(tmp_id)

        self.event_pos_ids = position_ids

    def clip_los(self, clip_days):
        if self.los > clip_days:
            self.los = clip_days

    def remove_indexes(self, indexes):
        self.event_tokens = np.delete(self.event_tokens, indexes)
        self.event_tokens_orig = np.delete(self.event_tokens_orig, indexes)
        self.event_times = np.delete(self.event_times, indexes)
        self.event_types = np.delete(self.event_types, indexes)
        self.event_values = np.delete(self.event_values, indexes)
