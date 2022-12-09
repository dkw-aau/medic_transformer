import math

import pandas as pd
import re


class DataImputer:
    def __init__(self):
        pass

    def min_max_impute_lab_tests(self, lab_tests, age, gender):
        npu_codes = ['NPU03429', 'NPU19673', 'NPU02319', 'NPU03230', 'NPU04169', 'NPU02593',
                     'NPU03568', 'NPU19651', 'NPU19748', 'NPU18016', 'NPU01370', 'NPU01944',
                     'NPU01459', 'DNK35131', 'NPU02902']

        npu_names = ['Natrium', 'Albumin', 'Haemoglobin', 'Kalium', 'Calcium', 'Leukocytter',
                     'Trombocytter', 'alat', 'CRP', 'Kreatinin', 'Bilirubin', 'Erythrocytter',
                     'Karbamid', 'Nyre-eGFR', 'Neutrofilocytter']

        features = []
        for npu_code in npu_codes:
            lab_rows = pd.DataFrame()

            # Extract valid npu measurements
            if not lab_tests.empty:
                lab_tests = lab_tests[lab_tests['value'].notna()]
                lab_rows = lab_tests.loc[lab_tests['npucode'] == npu_code]

            value_min, value_max = None, None

            # Extract min_max_values
            if not lab_rows.empty:
                value_min = lab_rows['value'].min()
                value_max = lab_rows['value'].max()

            ref_min, ref_max = self.lab_test_reference_values(npu_code, age, gender)

            if value_min and value_min < ref_min:
                ref_min = value_min
            if value_max and value_max > ref_max:
                ref_max = value_max

            features.append(ref_min)
            features.append(ref_max)

        feat_names = [item for sublist in [[code + '_min', code + '_max'] for code in npu_names] for item in sublist]
        return features, feat_names

    def min_max_impute_vitals(self, vitals):
        vital_codes = ['RNI_TEMP', 'RNI_RESPIRATION', 'RNI_ILTSATURATION', 'RNSR_BTSYS', 'RNSR_BTDIA', 'RNI_ILTTILSKUD']
        # ['RNI_BMI', 'RNI_PULS', 'RNI_MOSTELLERBAS']
        features = []

        for vit_code in vital_codes:
            vit_rows = pd.DataFrame()

            # Extract vital rows
            if not vitals.empty:
                vitals = vitals[vitals['value'].notna()]
                vit_rows = vitals.loc[vitals['intervention_code'] == vit_code]

            value_min, value_max = None, None

            # Extract min_max_values
            if not vit_rows.empty:
                value_min = vit_rows['value'].min()
                value_max = vit_rows['value'].max()

            ref_min, ref_max = self.vital_reference_values(vit_code)

            if value_min and value_min < ref_min:
                ref_min = value_min
            if value_max and value_max > ref_max:
                ref_max = value_max

            features.append(ref_min)
            features.append(ref_max)

        feat_names = [item for sublist in [[code + '_min', code + '_max'] for code in vital_codes] for item in sublist]

        return features, feat_names

    def impute_prescriptions(self, prescriptions):
        atc_codes = ['A', 'B', 'C', 'D', 'G', 'H', 'J', 'L', 'M', 'N', 'P', 'R', 'S', 'V']
        features = []

        prescs = []
        if not prescriptions.empty:
            prescs = list(set([x[0] for x in prescriptions['atc_recepter'].to_numpy()]))

        for code in atc_codes:
            if code in prescs:
                features.append(1)
            else:
                features.append(0)

        return features, atc_codes

    def lab_test_reference_values(self, npu_code, age, gender):
        if npu_code == 'NPU03429':
            min_value, max_value = 141, 141
        elif npu_code == 'NPU19673':
            if age <= 39:
                min_value, max_value = 42, 42
            elif age <= 69:
                min_value, max_value = 40.5, 40.5
            else:
                min_value, max_value = 39.5, 39.5
        elif npu_code == 'NPU02319':
            if gender == 'M':
                min_value, max_value = 9.4, 9.4
            else:
                min_value, max_value = 8.4, 8.4
        elif npu_code == 'NPU03230':
            min_value, max_value = 3.95, 3.95
        elif npu_code == 'NPU04169':
            min_value, max_value = 2.375, 2.375
        elif npu_code == 'NPU02593':
            min_value, max_value = 6.15, 6.15
        elif npu_code == 'NPU03568':
            if gender == 'M':
                min_value, max_value = 247.5, 247.5
            else:
                min_value, max_value = 277.5, 277.5
        elif npu_code == 'NPU19651':
            if gender == 'M':
                min_value, max_value = 40, 40
            else:
                min_value, max_value = 27.5, 27.5
        elif npu_code == 'NPU19748':
            min_value, max_value = 0.5, 0.5
        elif npu_code == 'NPU18016':
            if gender == 'M':
                min_value, max_value = 82.5, 82.5
            else:
                min_value, max_value = 67.5, 67.5
        elif npu_code == 'NPU01370':
            min_value, max_value = 15, 15
        elif npu_code == 'NPU01944':
            min_value, max_value = 92.5, 92.5
        elif npu_code == 'NPU01459':
            if gender == 'M' and age <= 49:
                min_value, max_value = 5.65, 5.65
            elif gender == 'M':
                min_value, max_value = 5.8, 5.8
            elif age <= 49:
                min_value, max_value = 3.9, 3.9
            else:
                min_value, max_value = 5.5, 5.5
        elif npu_code == 'DNK35131':
            min_value, max_value = 60, 60
        elif npu_code == 'NPU02902':
            min_value, max_value = 3.75, 3.75
        elif npu_code == 'NPU01443':
            min_value, max_value = 2.375, 2.375
        elif npu_code == 'DNK35302':
            min_value, max_value = 60, 60
        else:
            return NotImplementedError

        return min_value, max_value

    def vital_reference_values(self, vit_code):
        if vit_code == 'RNI_TEMP':
            min_value, max_value = 37, 37
        elif vit_code == 'RNI_ILTSATURATION':
            min_value, max_value = 100, 100
        elif vit_code == 'RNI_BMI':
            min_value, max_value = 23, 23
        elif vit_code == 'RNSR_BTSYS':
            min_value, max_value = 120, 120
        elif vit_code == 'RNSR_BTDIA':
            min_value, max_value = 80, 80
        elif vit_code == 'RNI_ILTTILSKUD':
            min_value, max_value = 0, 0
        elif vit_code == 'RNI_RESPIRATION':
            min_value, max_value = 12, 16
        else:
            return NotImplementedError

        return min_value, max_value
