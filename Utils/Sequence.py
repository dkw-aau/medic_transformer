import datetime
import time
from bisect import bisect

from Utils.Tokenizer import Tokenizer
import pandas as pd
pd.options.mode.chained_assignment = None


class Sequence:
    def __init__(self,
                 pat_id: int,
                 apriori: list,
                 apriori_names: list,
                 pd_lab: pd.DataFrame,
                 pd_vital: pd.DataFrame,
                 pd_diag: pd.DataFrame,
                 pd_medi: pd.DataFrame,
                 ed_start: pd.Timestamp,
                 ed_end: pd.Timestamp,
                 hosp_end: pd.Timestamp,
                 ed_length_of_stay: float,
                 hosp_length_of_stay: float,
                 length_of_stay: float,
                 mortality_30: bool,
                 req_hosp: bool,
                 uns_diag: bool):
        self.pat_id = pat_id
        self.apriori = apriori,
        self.apriori_names = apriori_names
        self.pd_lab = pd_lab
        self.pd_vital = pd_vital
        self.pd_diag = pd_diag
        self.pd_medi = pd_medi
        self.ed_start = ed_start
        self.ed_end = ed_end
        self.hosp_end = hosp_end
        self.ed_length_of_stay = ed_length_of_stay
        self.hosp_length_of_stay = hosp_length_of_stay
        self.length_of_stay = length_of_stay
        self.mortality_30 = mortality_30
        self.req_hosp = req_hosp
        self.uns_diag = uns_diag
        self.event_pos_ids = None
        self.event_times = None
        self.event_tokens = None
        self.event_tokens_orig = None
        self.event_types = None
        self.event_values = None
        self.token_vital = None
        self.token_diag = None
        self.token_lab = None
        self.token_apriori = None
        self.apriori_len = None
        self.token_value_dict = None

        # Imputer
        self.tokenizer = Tokenizer()

    def tokenize_vitals(self):
        self.token_vital = self.tokenizer.tokenize_vitals(self.pd_vital)

    def tokenize_diagnoses(self):
        self.token_diag = self.tokenizer.tokenize_diagnoses(self.pd_diag)

    def tokenize_labs(self):
        self.token_lab = self.tokenizer.tokenize_labs(self.pd_lab)

    def tokenize_apriori(self):
        self.token_apriori = self.tokenizer.tokenize_apriori(
            apriori=self.apriori[0],
            age_groups=10,
            names=self.apriori_names,
            ed_start=self.ed_start)
        self.apriori_len = len(self.token_apriori)

    def tokenize_all(self):
        self.tokenize_apriori()
        self.tokenize_labs()
        self.tokenize_vitals()
        self.tokenize_diagnoses()

    def create_patient_string(self, types):
        pd_combined = pd.DataFrame({'token': [], 'token_orig': [], 'event_time': [], 'event_type': [], 'event_value': []})
        data_types = [('diag', self.token_diag), ('lab', self.token_lab), ('vital', self.token_vital)]
        for d_type, data in data_types:
            if d_type in types and data is not None:
                data['event_type'] = d_type
                pd_combined = pd.concat([pd_combined, data])

        sorted_data = pd_combined.sort_values(by=['event_time'])

        # Add apriori knowledge
        if 'apriori' in types:
            self.token_apriori['event_type'] = 'apriori'
            sorted_data = pd.concat([self.token_apriori, sorted_data])

        self.event_tokens = sorted_data['token'].tolist()
        self.event_tokens_orig = sorted_data['token_orig'].tolist()
        self.event_times = sorted_data['event_time'].tolist()
        self.event_types = sorted_data['event_type'].tolist()
        self.event_values = sorted_data['event_value'].tolist()
        self.create_position_ids()

    def create_position_ids(self):
        tmp_id = 0
        tmp_time = self.event_times[0]
        segment_dist = 120
        position_ids = []

        for event_time in self.event_times:
            if event_time < tmp_time + datetime.timedelta(seconds=segment_dist):
                position_ids.append(tmp_id)
            else:
                tmp_id += 1
                tmp_time = event_time
                position_ids.append(tmp_id)

        self.event_pos_ids = position_ids

    def cut_by_hours(self, cutoff_hours):
        end_time = self.ed_start + datetime.timedelta(hours=cutoff_hours, seconds=1)
        index = bisect(self.event_times, end_time)
        self.event_times = self.event_times[0:index]
        self.event_tokens_orig = self.event_tokens_orig[0:index]
        self.event_values = self.event_values[0:index]
        self.event_tokens = self.event_tokens[0:index]
        self.event_types = self.event_types[0:index]
        self.event_pos_ids = self.event_pos_ids[0:index]

    def group_token_values(self, tokens):
        token_value_dict = {}
        for token, value in zip(self.event_tokens_orig, self.event_values):
            if token in tokens:
                if token in token_value_dict:
                    token_value_dict[token].append(value)
                else:
                    token_value_dict[token] = [value]

        self.token_value_dict = token_value_dict

    def minimize(self):
        self.apriori = None
        self.apriori_names = None
        self.pd_lab = None
        self.pd_vital = None
        self.pd_diag = None
        self.pd_medi = None
        self.token_vital = None
        self.token_diag = None
        self.token_lab = None
        self.token_apriori = None
