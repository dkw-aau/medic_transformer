from datetime import datetime

import pandas as pd
from DataImputer import DataImputer
from datetime import timedelta


class Forloeb:
    def __init__(self):
        self.start_ska = None
        self.end_ska = None
        self.end_hosp = None
        self.age = None
        self.ccm_score = None
        self.triage_kat = None
        self.mortality_30 = None
        self.last_diagnosis = None
        self.gender = None
        self.ankomst_dag = None
        self.ankomst_aften = None
        self.ankomst_nat = None
        self.ankomst_weekend = None
        self.ankomst_hverdag = None
        self.ambulance = None
        self.vitals = None
        self.diagnosis_codes = None
        self.lab_tests = None
        self.presc = None
        self.imputer = DataImputer()
        self.komorb_string = None

    def set_comorbs(self, comorb):
        if comorb:
            self.create_comorbidities(comorb)
        else:
            self.create_comorbidities('0' * 32)

    def set_ankomst(self, arrival_day):
        if arrival_day in [1, 2, 3, 4, 5]:
            self.ankomst_weekend = 1
            self.ankomst_hverdag = 0
        else:
            self.ankomst_weekend = 0
            self.ankomst_hverdag = 1

    def set_ambulance(self, driving_type):
        if driving_type in ['Normal', 'WithSiren']:
            self.ambulance = 1
        else:
            self.ambulance = 0

    def get_pre_knowledge(self):
        comorb, comorb_names = self.get_comorbidities()
        presc, presc_names = self.get_prescriptions()
        times, time_names = self.get_time_features()
        others, other_names = self.get_other_features()

    def get_time_features(self):
        time_features = [self.ankomst_dag, self.ankomst_aften, self.ankomst_nat,
                         self.ankomst_weekend, self.ankomst_hverdag]

        time_names = ['ankomst_dag', 'ankomst_aften', 'ankomst_nat', 'ankomst_weekend', 'ankomst_hverdag']

        return time_features, time_names

    def get_other_features(self):
        other_features = [self.age, self.ccm_score, self.triage_kat, self.gender, self.ambulance]

        other_names = ['age', 'ccm_score', 'triage_kat', 'gender', 'ambulance']

        return other_features, other_names

    def get_prescriptions(self):
        atc_codes = ['A', 'B', 'C', 'D', 'G', 'H', 'J', 'L', 'M', 'N', 'P', 'R', 'S', 'V']
        features = []

        if self.presc:
            prescs = list(set([x[0] for x in self.presc['atc_recepter'].to_numpy()]))

            for code in atc_codes:
                if code in prescs:
                    features.append(1)
                else:
                    features.append(0)
        else:
            features = [0] * len(atc_codes)

        return features, atc_codes

    def get_comorbidities(self):
        comorbidities = [self.acute_myocardial_infarction, self.congestive_heart_failure,
                         self.peripheral_vascular_disease, self.cerebral_vascular_accident,
                         self.dementia, self.pulmonary_disease, self.connective_tissue_disorder,
                         self.peptic_ulcer, self.liver_disease, self.diabetes, self.diabetes_complications,
                         self.paraplegia, self.renal_disease, self.cancer, self.leukemia, self.lymphoma,
                         self.severe_liver_disease, self.metastatic_cancer]

        names = ['acute_myocardial_infarction', 'congestive_heart_failure',
                 'peripheral_vascular_disease', 'cerebral_vascular_accident',
                 'dementia', 'pulmonary_disease', 'connective_tissue_disorder',
                 'peptic_ulcer', 'liver_disease', 'diabetes', 'diabetes_complications',
                 'paraplegia', 'renal_disease', 'cancer', 'leukemia', 'lymphoma',
                 'severe_liver_disease', 'metastatic_cancer']

        return comorbidities, names

    def create_comorbidities(self, comorb):
        self.acute_myocardial_infarction = 1 if comorb[31:32] == '1' else 0
        self.congestive_heart_failure = 1 if comorb[30:31] == '1' else 0
        self.peripheral_vascular_disease = 1 if comorb[29:30] == '1' else 0
        self.cerebral_vascular_accident = 1 if comorb[28:29] == '1' else 0
        self.dementia = 1 if comorb[27:28] == '1' else 0
        self.pulmonary_disease = 1 if comorb[26:27] == '1' else 0
        self.connective_tissue_disorder = 1 if comorb[25:26] == '1' else 0
        self.peptic_ulcer = 1 if comorb[24:25] == '1' else 0
        self.liver_disease = 1 if comorb[23:24] == '1' else 0
        self.diabetes = 1 if comorb[22:23] == '1' else 0
        self.diabetes_complications = 1 if comorb[21:22] == '1' else 0
        self.paraplegia = 1 if comorb[20:21] == '1' else 0
        self.renal_disease = 1 if comorb[19:20] == '1' else 0
        self.cancer = 1 if comorb[18:19] == '1' else 0
        self.leukemia = 1 if comorb[17:18] == '1' else 0
        self.lymphoma = 1 if comorb[16:17] == '1' else 0
        self.severe_liver_disease = 1 if comorb[15:16] == '1' else 0
        self.metastatic_cancer = 1 if comorb[14:15] == '1' else 0

    def set_gender(self, koen):
        if koen == 'M':
            self.gender = 1
        else:
            self.gender = 0

    def set_lab_tests(self, lab_tests):
        self.lab_tests = lab_tests

    def set_diagnosis(self, diagnosis_codes):
        self.diagnosis_codes = diagnosis_codes

    def get_lab_tests_by_time(self, start, end):
        lab_tests = pd.DataFrame()
        if self.lab_tests is not None:
            start_index = self.lab_tests['timestamp'].searchsorted(start)
            end_index = self.lab_tests['timestamp'].searchsorted(end)
            lab_tests = self.lab_tests.reset_index(drop=True).loc[start_index: end_index - 1]
        return lab_tests

    def get_vitals_by_time(self, start, end):
        vitals = pd.DataFrame()
        if self.vitals is not None:
            start_index = self.vitals['timestamp'].searchsorted(start)
            end_index = self.vitals['timestamp'].searchsorted(end)
            vitals = self.vitals.reset_index(drop=True).loc[start_index: end_index - 1]
        return vitals

    def get_prescriptions_by_time(self, start, end):
        presc = pd.DataFrame()
        if self.presc is not None:
            start_index = self.presc['timestamp'].searchsorted(start)
            end_index = self.presc['timestamp'].searchsorted(end)
            presc = self.presc.reset_index(drop=True).loc[start_index: end_index - 1]
        return presc

    def set_vitals(self, vitals):
        self.vitals = vitals

    def set_presc(self, presc):
        self.presc = presc

    def get_samples_hourly_split(self, num_buckets, test_start):
        # is train or test
        if self.start_ska > test_start:
            print('Test Patient')
        else:
            print("Train Patient")

        return [None], None

    def get_counts(self):
        start = self.start_ska

        if self.end_ska == self.end_hosp:
            patient_type = 'ED'
        else:
            patient_type = 'HOSP'

        # Get bucket values for lab_tests
        lab_tests_ed = self.get_lab_tests_by_time(start, self.end_ska)
        vitals_ed = self.get_vitals_by_time(start, self.end_ska)
        presc = self.get_prescriptions_by_time(start - timedelta(days=730), self.end_ska)

        lab_tests_hosp, vitals_hosp, presc_hosp = [], [], []
        if patient_type == 'HOSP':
            lab_tests_hosp = self.get_lab_tests_by_time(start, self.end_hosp)
            vitals_hosp = self.get_vitals_by_time(start, self.end_hosp)

        return lab_tests_ed, lab_tests_hosp, vitals_ed, vitals_hosp, presc, patient_type

    def get_sample_fam_end(self):
        return self.get_samples_by_buckets(1)

    def get_sample_fam_begin(self):
        found_presc = pd.DataFrame()

        hosp_days_remaining = (self.end_hosp - self.end_ska).total_seconds() / 60 / 60 / 24
        required_hosp = 0 if self.end_hosp == self.end_ska else 1
        gender = 1 if self.gender == 'M' else 0

        # Get bucket values for prescriptions
        if self.presc is not None:
            found_presc = self.presc

        sample, feat_names = self.create_sample(presc=found_presc)

        # Add extra patient info to sample
        patient_feats = [self.age, gender, self.ccm_score, self.triage_kat, self.ankomst_dag,
                         self.ankomst_aften, self.ankomst_nat, self.ankomst_weekend, self.ankomst_hverdag,
                         self.ambulance, required_hosp, hosp_days_remaining]
        patient_feat_names = ['age', 'gender', 'ccm_score', 'triage_kat', 'ankomst_dag', 'ankomst_aften',
                              'ankomst_nat', 'ankomst_weekend', 'ankomst_hverdag', 'ambulance', 'required_hosp',
                              'hosp_time_remaining']

        full_sample = sample + patient_feats
        full_feat_names = feat_names + patient_feat_names
        full_feat_names = [feat.lower() for feat in full_feat_names]

        return [full_sample], full_feat_names

    def get_samples_by_buckets(self, num_buckets, end_time='ska'):
        if end_time == 'ska':
            end_range = self.end_ska
        elif end_time == 'hosp':
            end_range = self.end_hosp
        else:
            return NotImplementedError

        # Get bucket ranges
        # For each bucket we create a patient sample
        ranges = self.date_ranges(self.start_ska, end_range, num_buckets)

        # Aggregate datafrrames
        found_lab_tests = pd.DataFrame()
        found_vitals = pd.DataFrame()
        found_presc = pd.DataFrame()

        samples = []
        for bucket_start, bucket_end in zip(ranges[0:-1], ranges[1:]):
            hours_since_ska = (bucket_end - self.start_ska).total_seconds() / 60 / 60
            hosp_days_remaining = (self.end_hosp - bucket_end).total_seconds() / 60 / 60 / 24
            required_hosp = 0 if self.end_hosp == self.end_ska else 1
            gender = 1 if self.gender == 'M' else 0

            # Get bucket values for lab_tests
            if self.lab_tests is not None:
                lab_start_index = self.lab_tests['timestamp'].searchsorted(bucket_start)
                lab_end_index = self.lab_tests['timestamp'].searchsorted(bucket_end)
                bucket_labs_tests = self.lab_tests.reset_index(drop=True).loc[lab_start_index: lab_end_index - 1]

                if not found_lab_tests.empty and bucket_labs_tests.empty:
                    found_lab_tests = pd.concat([found_lab_tests, bucket_labs_tests])
                elif not bucket_labs_tests.empty and found_lab_tests.empty:
                    found_lab_tests = bucket_labs_tests

            # Get bucket values for vital_tests
            if self.vitals is not None:
                vital_start_index = self.vitals['timestamp'].searchsorted(bucket_start)
                vital_end_index = self.vitals['timestamp'].searchsorted(bucket_end)
                bucket_vitals = self.vitals.reset_index(drop=True).loc[vital_start_index: vital_end_index - 1]

                if not found_vitals.empty and bucket_vitals.empty:
                    found_vitals = pd.concat([found_vitals, bucket_vitals])
                elif not bucket_vitals.empty and found_vitals.empty:
                    found_vitals = bucket_vitals

            # Get bucket values for prescriptions
            if self.presc is not None:
                found_presc = self.presc

            sample, feat_names = self.create_sample(found_lab_tests, found_vitals, found_presc)

            # Add extra patient info to sample
            extra_feats = [self.age, gender, self.ccm_score, self.triage_kat, self.ankomst_dag, self.ankomst_aften,
                           self.ankomst_nat, self.ankomst_weekend, self.ankomst_hverdag, self.ambulance,
                           hours_since_ska, self.last_diagnosis, required_hosp, hosp_days_remaining, self.mortality_30]
            #count_feats = [found_lab_tests.shape[0], found_vitals.shape[0], found_presc.shape[0]]

            extra_feat_names = ['age', 'gender', 'ccm_score', 'triage_kat', 'ankomst_dag', 'ankomst_aften',
                                'ankomst_nat', 'ankomst_weekend', 'ankomst_hverdag', 'ambulance', 'hours_since_ska',
                                'last_diagnosis', 'required_hosp', 'hosp_time_remaining', 'mortality_30']
            #count_feat_names = ['num_lab_tests', 'num_vitals', 'num_presc_cats']

            full_sample = sample + extra_feats
            full_feat_names = feat_names + extra_feat_names
            full_feat_names = [feat.lower() for feat in full_feat_names]
            samples.append(full_sample)

        return samples, full_feat_names

    def date_ranges(self, start, end, intv):
        diff = (end - start) / intv
        return [start + diff * i for i in range(intv)] + [end]

    def create_sample(self, lab_tests=None, vitals=None, presc=None):
        names, sample = [], []

        if lab_tests is not None:
            lab_features, npu_codes = self.imputer.min_max_impute_lab_tests(lab_tests, self.age, self.gender)
            names = names + npu_codes
            sample = sample + lab_features
        if vitals is not None:
            vital_features, vital_codes = self.imputer.min_max_impute_vitals(vitals)
            names = names + vital_codes
            sample = sample + vital_features
        if presc is not None:
            presc_features, atc_codes = self.imputer.impute_prescriptions(presc)
            names = names + atc_codes
            sample = sample + presc_features

        return sample, names




