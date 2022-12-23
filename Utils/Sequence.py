from Utils.Tokenizer import Tokenizer
import pandas as pd


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
        self.length_of_stay = length_of_stay
        self.mortality_30 = mortality_30
        self.req_hosp = req_hosp
        self.uns_diag = uns_diag
        self.event_times = None
        self.event_tokens = None
        self.token_vital = None
        self.token_diag = None
        self.token_lab = None
        self.token_apriori = None
        self.apriori_len = None

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
        pd_combined = pd.DataFrame({'token': [], 'event_time': []})
        data_types = [('diag', self.token_diag), ('lab', self.token_lab), ('vital', self.token_vital)]
        for d_type, data in data_types:
            if d_type in types and data is not None:
                pd_combined = pd.concat([pd_combined, data])

        sorted_data = pd_combined.sort_values(by=['event_time'])

        # Add apriori knowledge
        if 'apriori' in types:
            sorted_data = pd.concat([self.token_apriori, sorted_data])

        self.event_tokens = sorted_data['token'].tolist()
        self.event_times = sorted_data['event_time'].tolist()

    def minimize(self):
        self.apriori = None,
        self.apriori_names = None
        self.pd_lab = None
        self.pd_vital = None
        self.pd_diag = None
        self.pd_medi = None
        self.token_vital = None
        self.token_diag = None
        self.token_lab = None
        self.token_apriori = None
