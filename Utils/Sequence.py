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
                 pd_adm: pd.DataFrame,
                 pd_procs: pd.DataFrame,
                 ed_start: pd.Timestamp,
                 ed_end: pd.Timestamp,
                 hosp_end: pd.Timestamp,
                 ed_length_of_stay: float,
                 hosp_length_of_stay: float,
                 length_of_stay: float,
                 mortality_30: bool,
                 req_hosp: bool,
                 uns_diag: bool,
                 age: int,
                 gender: int):
        self.pat_id = pat_id
        self.apriori = apriori
        self.apriori_names = apriori_names
        self.pd_lab = pd_lab
        self.pd_vital = pd_vital
        self.pd_diag = pd_diag
        self.pd_adm = pd_adm
        self.pd_procs = pd_procs
        self.ed_start = ed_start
        self.ed_end = ed_end
        self.hosp_end = hosp_end
        self.ed_length_of_stay = ed_length_of_stay
        self.hosp_length_of_stay = hosp_length_of_stay
        self.length_of_stay = length_of_stay
        self.mortality_30 = mortality_30
        self.req_hosp = req_hosp
        self.uns_diag = uns_diag
        self.age = age
        self.gender = gender
        self.event_pos_ids = None
        self.event_times = None
        self.event_tokens = None
        self.event_tokens_orig = None
        self.event_types = None
        self.event_values = None
        self.token_vital = None
        self.token_diag = None
        self.token_lab = None
        self.token_adm = None
        self.token_procs = None
        self.token_apriori = None
        self.apriori_len = None
        self.token_value_dict = None
        self.label = None

        # Imputer
        self.tokenizer = Tokenizer()

    def tokenize_vitals(self):
        self.token_vital = self.tokenizer.tokenize_vitals(self.pd_vital)

    def tokenize_diagnoses(self):
        self.token_diag = self.tokenizer.tokenize_diagnoses(self.pd_diag)

    def tokenize_adm(self):
        self.token_adm = self.tokenizer.tokenize_adm(self.pd_adm)

    def tokenize_procs(self):
        self.token_procs = self.tokenizer.tokenize_procs(self.pd_procs)

    def tokenize_labs(self):
        self.token_lab = self.tokenizer.tokenize_labs(self.pd_lab)

    def tokenize_apriori(self):
        self.token_apriori = self.tokenizer.tokenize_apriori(
            apriori=self.apriori,
            names=self.apriori_names,
            ed_start=self.ed_start)
        self.apriori_len = len(self.token_apriori)

    def tokenize_all(self):
        self.tokenize_apriori()
        self.tokenize_labs()
        self.tokenize_vitals()
        self.tokenize_diagnoses()
        self.tokenize_adm()
        self.tokenize_procs()

    def create_patient_string(self, types):
        pd_combined = pd.DataFrame({'token': [], 'token_orig': [], 'event_time': [], 'event_type': [], 'event_value': []})
        data_types = [
            ('diag', self.token_diag),
            ('lab', self.token_lab),
            ('vital', self.token_vital),
            ('adm', self.token_adm),
            ('proc', self.token_procs)]

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

    def create_position_ids(self, dist=60):
        tmp_id = 0
        tmp_time = self.ed_start
        position_ids = []

        for event_time in self.event_times:
            if event_time < tmp_time + datetime.timedelta(seconds=dist):
                position_ids.append(tmp_id)
            else:
                tmp_id += 1
                tmp_time = event_time
                position_ids.append(tmp_id)

        self.event_pos_ids = position_ids

    def cut_by_hours(self, cutoff_hours=None):
        if cutoff_hours is not None and cutoff_hours >= 0:
            end_time = self.ed_start + datetime.timedelta(hours=cutoff_hours, seconds=1)
            index = bisect(self.event_times, end_time)
            self.event_times = self.event_times[0:index]
            self.event_tokens_orig = self.event_tokens_orig[0:index]
            self.event_values = self.event_values[0:index]
            self.event_tokens = self.event_tokens[0:index]
            self.event_types = self.event_types[0:index]
            self.event_pos_ids = self.event_pos_ids[0:index]

    def minus_los(self, hours):
        self.length_of_stay = (self.length_of_stay * 24 - hours) / 24

    def group_token_values(self):
        token_value_dict = {}
        for token, value in zip(reversed(self.event_tokens_orig), reversed(self.event_values)):
            if token in token_value_dict:
                continue
            else:
                token_value_dict[token] = value

        # Add age and gender
        token_value_dict['age'] = self.age
        token_value_dict['gender'] = self.gender
        self.token_value_dict = token_value_dict

    def create_label(self, conf):
        lab = None
        if 'real' == conf['task']:
            lab = self.length_of_stay
        elif 'binary' == conf['task']:
            lab = 1 if self.length_of_stay > conf['binary_thresh'] else 0
        elif 'category' == conf['task']:
            lab = bisect(conf['cats'], self.length_of_stay)
        elif conf['task'] == 'm30':
            lab = 1 if self.mortality_30 else 0
        elif conf['task'] == 'mlm':
            lab = None
        else:
            exit(f'Task: {conf["task"]} not implemented for sequence')

        self.label = lab

    def clip_los(self, clip_days):
        if self.length_of_stay > clip_days:
            self.length_of_stay = clip_days

    def remove_indexes(self, indexes):
        for index in sorted(indexes, reverse=True):
            del self.event_tokens[index]
            del self.event_tokens_orig[index]
            del self.event_times[index]
            del self.event_types[index]
            del self.event_values[index]

    def minimize(self):
        self.apriori = None
        self.apriori_names = None
        self.pd_lab = None
        self.pd_vital = None
        self.pd_diag = None
        self.pd_adm = None
        self.pd_procs = None
        self.token_vital = None
        self.token_diag = None
        self.token_lab = None
        self.token_apriori = None
