import numpy
import pandas as pd
import math


class Tokenizer:
    not_found = 0

    def __init__(self):
        pass

    def tokenize_labs(self, labs):
        if labs is not None and not labs.empty:
            labs['token_orig'] = labs['npucode']
            labs['token'] = labs.apply(self.lab_reference_values, axis=1)
            labs.dropna(inplace=True)
            labs['event_value'] = labs['value']
            return labs[['token', 'token_orig', 'event_value', 'event_time']]
        else:
            return None

    def tokenize_vitals(self, vitals):
        if vitals is not None and not vitals.empty:
            vitals['token_orig'] = vitals['intervention_code']
            vitals['token'] = vitals.apply(self.vital_reference_values, axis=1)
            vitals.dropna(inplace=True)
            vitals['event_value'] = vitals['value']
            return vitals[['token', 'token_orig', 'event_value', 'event_time']]
        else:
            return None

    def tokenize_diagnoses(self, diagnoses):
        if diagnoses is not None and not diagnoses.empty:
            diagnoses.drop_duplicates(inplace=True)
            diagnoses['token_orig'] = diagnoses['diakod']
            diagnoses['token'] = diagnoses['diakod']
            diagnoses['event_value'] = 1
            return diagnoses[['token', 'token_orig', 'event_value', 'event_time']]
        else:
            return None

    def tokenize_adm(self, adm):
        if adm is not None and not adm.empty:
            adm.drop_duplicates(inplace=True)
            adm['token_orig'] = adm['event_code']
            adm['token'] = adm['event_code']
            adm['event_value'] = 1
            return adm[['token', 'token_orig', 'event_value', 'event_time']]
        else:
            return None

    def tokenize_procs(self, procs):
        if procs is not None and not procs.empty:
            procs.drop_duplicates(inplace=True)
            procs['token_orig'] = procs['event_code']
            procs['token'] = procs['event_code']
            procs['event_value'] = 1
            return procs[['token', 'token_orig', 'event_value', 'event_time']]
        else:
            return None

    def tokenize_apriori(self, apriori, names, ed_start):
        diseases = [f'{name}_{"pos" if val == 1 else "neg"}' for name, val in
                    zip(names[0:18], apriori[0:18])]
        prescs = [f'{name}_{"pos" if val == 1 else "neg"}' for name, val in
                  zip(names[18:32], apriori[18:32])]
        ambulance = [f'amb_{"pos" if apriori[32] == 1 else "neg"}']
        triage_kat = [f'triage_{int(apriori[33])}' if int(apriori[33]) != -1 else 'UNK']
        arrival_weekend = [f'arr_weekend_{"pos" if apriori[34] == 1 else "neg"}']
        arrival_day = [f'arr_day_{apriori[35]}']
        arrival_evening = [f'arr_evening_{"pos" if apriori[36] == 1 else "neg"}']
        arrival_night = [f'arr_night_{"pos" if apriori[37] == 1 else "neg"}']

        tokens = ['CLS'] + diseases + prescs + ambulance + triage_kat
        tokens = tokens + arrival_weekend + arrival_day + arrival_evening + arrival_night
        values = [1] + apriori
        token_orig = ['CLS'] + names

        pd_apriori = pd.DataFrame({'token': tokens, 'token_orig': token_orig, 'event_value': values, 'event_time': [ed_start] * len(tokens)})

        return pd_apriori

    def lab_reference_values(self, row):
        reftext = row[3]
        if reftext is None:
            Tokenizer.not_found += 1
            return numpy.nan
        if 'Ikke relevant' in reftext:
            Tokenizer.not_found += 1
            return numpy.nan
        if 'Ikke fastlagt' in reftext:
            Tokenizer.not_found += 1
            return numpy.nan

        if '<' in reftext:
            refnumber = float(reftext.replace('<', '').replace(',', '.'))
            if row[2] > refnumber:
                return row[1] + '_high'
            else:
                return row[1] + '_normal'
        elif '>' in reftext:
            refnumber = float(reftext.replace('>', '').replace(',', '.'))
            if row[2] < refnumber:
                return row[1] + '_low'
            else:
                return row[1] + '_normal'
        elif '-' in reftext:
            occ = reftext.count('-')
            if occ == 1:
                low, high = reftext.replace(',', '.').split('-')
            else:
                idx2 = reftext.index('-', 1)
                low = reftext[:idx2]
                high = reftext[idx2 + 1:]
            if row[2] < float(low):
                return row[1] + '_low'
            elif row[2] > float(high):
                return row[1] + '_high'
            else:
                return row[1] + '_normal'
        else:
            NotImplementedError

    def vital_reference_values(self, row):
        if row[1] == 'RNI_TEMP':
            if row[2] <= 36.6:
                return 'temp_low'
            elif row[2] >= 38:
                return 'temp_high'
            else:
                return 'temp_normal'
        elif row[1] == 'RNI_ILTSATURATION':
            if row[2] < 100:
                return 'ilt_low'
            else:
                return 'ilt_normal'
        elif row[1] == 'RNI_BMI':
            if row[2] < 18.5:
                return 'bmi_low'
            elif row[2] > 24.9:
                return 'bmi_high'
            else:
                return 'bmi_normal'
        elif row[1] == 'RNSR_BTSYS':
            if row[2] < 90:
                return 'systolic_low'
            elif row[2] > 140:
                return 'systolic_high'
            else:
                return 'systolic_normal'
        elif row[1] == 'RNSR_BTDIA':
            if row[2] < 70:
                return 'diastolic_low'
            elif row[2] > 90:
                return 'diastolic_high'
            else:
                return 'diastolic_normal'
        elif row[1] == 'RNI_ILTTILSKUD':
            if row[2] <= 2:
                return 'ilttilskud_normal'
            else:
                return 'ilttilskud_high'
        elif row[1] == 'RNI_RESPIRATION':
            if row[2] < 12:
                return 'respiration_low'
            elif row[2] > 18:
                return 'respiration_high'
            else:
                return 'respiration_normal'
        elif row[1] == 'RNI_PULS':
            if row[2] < 50:
                return 'puls_low'
            elif row[2] > 90:
                return 'puls_high'
            else:
                return 'puls_normal'
        else:
            return NotImplementedError
