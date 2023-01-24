import math

import pandas as pd
from datetime import timedelta
from tqdm import tqdm
import os

from datasetmodule.Tokenizer import Tokenizer


def get_is_unspecific(diagnosis):
    if 'DR' in diagnosis or 'DZ' in diagnosis:
        return True
    else:
        return False


def get_demographics(gender, age):
    feats = [1 if gender == 'M' else 0, int(age)]
    names = ['gender', 'age']

    return feats, names


def get_arr_time(arr_day, day, evening, night):
    feats = []
    names = ['arrival_weekend', 'arrival_day', 'arrival_evening', 'arrival_night']

    if arr_day in [1, 2, 3, 4, 5]:
        feats.append(0)
    else:
        feats.append(1)

    feats.append(int(day))
    feats.append(int(evening))
    feats.append(int(night))

    return feats, names


def get_comorbidities(comorb):
    if not comorb:
        comorb = '0' * 32

    feats = []
    for index in range(32, 14, -1):
        feats.append(1 if comorb[index - 1:index] == '1' else 0)

    names = ['acute_myocardial_infarction', 'congestive_heart_failure',
             'peripheral_vascular_disease', 'cerebral_vascular_accident',
             'dementia', 'pulmonary_disease', 'connective_tissue_disorder',
             'peptic_ulcer', 'liver_disease', 'diabetes', 'diabetes_complications',
             'paraplegia', 'renal_disease', 'cancer', 'leukemia', 'lymphoma',
             'severe_liver_disease', 'metastatic_cancer']

    return feats, names


def get_extra(driving, triage):
    names = ['ambulance', 'triage_kat']
    feats = [1 if driving is not None else 0, -1 if math.isnan(triage) else triage]

    return feats, names


def get_prescriptions(presc):
    names = ['A', 'B', 'C', 'D', 'G', 'H', 'J', 'L', 'M', 'N', 'P', 'R', 'S', 'V']
    feats = []

    if presc is not None and not presc.empty:
        prescs = list(set([x[0] for x in presc['atc_recept'].to_numpy()]))

        for code in names:
            if code in prescs:
                feats.append(1)
            else:
                feats.append(0)
    else:
        feats = [0] * len(names)

    return feats, names


def prepare_typed_data(group_data, patient_id, start, end, labels):
    if patient_id in group_data.groups.keys():
        patient_data = group_data.get_group(patient_id)

        # Find values from forloeb
        first_index = patient_data['event_time'].searchsorted(start)
        last_index = patient_data['event_time'].searchsorted(end)

        forloeb_data = patient_data.reset_index().loc[first_index: last_index - 1]
        forloeb_data = forloeb_data[labels]

        return forloeb_data
    else:
        return None


def tokenize_vitals(tokenizer, pd_vital):
    return tokenizer.tokenize_vitals(pd_vital)


def tokenize_diagnoses(tokenizer, pd_diag):
    return tokenizer.tokenize_diagnoses(pd_diag)


def tokenize_adm(tokenizer, pd_adm):
    return tokenizer.tokenize_adm(pd_adm)


def tokenize_procs(tokenizer, pd_procs):
    return tokenizer.tokenize_procs(pd_procs)


def tokenize_labs(tokenizer, pd_lab):
    return tokenizer.tokenize_labs(pd_lab)


def tokenize_apriori(tokenizer, apriori, apriori_names, ed_start):
    token_apriori = tokenizer.tokenize_apriori(
        apriori=apriori,
        names=apriori_names,
        ed_start=ed_start)
    return token_apriori


def create_patient_string(tok_apriori, tok_diag, tok_lab, tok_vital, tok_adm, tok_proc):
    pd_combined = pd.DataFrame({'token': [], 'token_orig': [], 'event_time': [], 'event_type': [], 'event_value': []})
    data_types = [
        ('apriori', tok_apriori),
        ('diag', tok_diag),
        ('lab', tok_lab),
        ('vital', tok_vital),
        ('adm', tok_adm),
        ('proc', tok_proc)]

    for d_type, data in data_types:
        data['event_type'] = d_type
        pd_combined = pd.concat([pd_combined, data])

    # Sort data
    sorted_data = pd_combined.sort_values(by=['event_time'])

    event_tok = sorted_data['token'].tolist()
    event_tok_orig = sorted_data['token_orig'].tolist()
    event_time = sorted_data['event_time'].tolist()
    event_type = sorted_data['event_type'].tolist()
    event_value = sorted_data['event_value'].tolist()

    return event_tok, event_tok_orig, event_time, event_type, event_value


def prepare_patients(max_sequences, corpus_file, data_path):
    # Load files
    forloeb_file = 'forloeb'
    diagnosis_file = 'diagnosis'
    labtest_file = 'labka_all'
    vitals_file = 'ccs'
    pres_file = 'luna'
    adm_file = 'adm'
    proc_file = 'proc'

    forloeb = pd.read_parquet(os.path.join(data_path, f'{forloeb_file}.parquet'))
    diagnosis = pd.read_parquet(os.path.join(data_path, f'{diagnosis_file}.parquet'))
    labtests = pd.read_parquet(os.path.join(data_path, f'{labtest_file}.parquet'))
    vitals = pd.read_parquet(os.path.join(data_path, f'{vitals_file}.parquet'))
    prescs = pd.read_parquet(os.path.join(data_path, f'{pres_file}.parquet'))
    procs = pd.read_parquet(os.path.join(data_path, f'{proc_file}.parquet'))
    adm = pd.read_parquet(os.path.join(data_path, f'{adm_file}.parquet'))

    # Sort on time for optimization
    diagnosis = diagnosis.sort_values(by=['event_time'])
    labtests = labtests.sort_values(by=['event_time'])
    vitals = vitals.sort_values(by=['event_time'])
    prescs = prescs.sort_values(by=['event_time'])
    procs = procs.sort_values(by=['event_time'])
    adm = adm.sort_values(by=['event_time'])

    # Groupby for optimization
    diagnosis = diagnosis.groupby(by='patientid')
    labtests = labtests.groupby(by='patientid')
    vitals = vitals.groupby(by='patientid')
    prescs = prescs.groupby(by='patientid')
    procs = procs.groupby(by='patientid')
    adm = adm.groupby(by='patientid')

    # For saving all forloeb
    all_data = pd.DataFrame({'token': [], 'token_orig': [], 'event_time': [], 'event_type': [], 'event_value': [], 'patient_id': []})
    df_patients = pd.DataFrame({'patient_id': [], 'age': [], 'sex': [], 'hosp_start': [], 'los': []})

    tokenizer = Tokenizer()

    index = 0
    for _, row in tqdm(forloeb.iterrows(), total=forloeb.shape[0]):

        # Create new forloeb
        pat_id = row['patientid']
        ed_start = row['ankomst_ska']
        hosp_end = row['afgang']

        # Demographic Patient Data
        demo, demo_names = get_demographics(row['Kon'], row['Alder'])

        # Admission Time Data
        adm_time, adm_time_names = get_arr_time(
            row['ankomst_ugedag'],
            row['ankomst_dag'],
            row['ankomst_aften'],
            row['ankomst_nat']
        )

        # Comorbidity Data
        comorb, comorb_names = get_comorbidities(row['komorb_sygd_bin'])

        # Extra Data
        extra, extra_names = get_extra(
            row['DrivingTypeToDeliveryText'],
            row['TriageKat']
        )

        # Prescription Data
        pd_presc = prepare_typed_data(prescs, pat_id, ed_start - timedelta(days=730), hosp_end, ['event_time', 'atc_recept'])
        presc, presc_names = get_prescriptions(pd_presc)

        apriori = comorb + presc + extra + adm_time
        apriori_names = comorb_names + presc_names + extra_names + adm_time_names
        length_of_stay = (hosp_end - ed_start).total_seconds() / 60 / 60 / 24

        # Prepare data types
        pd_lab = prepare_typed_data(labtests, pat_id, ed_start, hosp_end, ['event_time', 'npucode', 'value', 'reftext'])
        pd_diag = prepare_typed_data(diagnosis, pat_id, ed_start, hosp_end, ['event_time', 'diakod'])
        pd_vitals = prepare_typed_data(vitals, pat_id, ed_start, hosp_end, ['event_time', 'intervention_code', 'value'])
        pd_adm = prepare_typed_data(adm, pat_id, ed_start, hosp_end, ['event_time', 'event_code'])
        pd_procs = prepare_typed_data(procs, pat_id, ed_start, hosp_end, ['event_time', 'event_code'])

        patient = {'patient_id': pat_id, 'age': demo[1], 'sex': demo[0], 'hosp_start': ed_start, 'los': length_of_stay}
        df_patients.append(patient, ignore_index=True)

        # Tokenize and create patient string
        tok_apriori = tokenize_apriori(tokenizer, apriori, apriori_names, ed_start)
        tok_lab = tokenize_labs(tokenizer, pd_lab)
        tok_vital = tokenize_vitals(tokenizer, pd_vitals)
        tok_diag = tokenize_diagnoses(tokenizer, pd_diag)
        tok_adm = tokenize_adm(tokenizer, pd_adm)
        tok_proc = tokenize_procs(tokenizer, pd_procs)

        event_tok, event_tok_orig, event_time, event_type, event_value = create_patient_string(tok_apriori, tok_diag, tok_lab, tok_vital, tok_adm, tok_proc)
        data = pd.DataFrame({'token': event_tok, 'token_orig': event_tok_orig, 'event_time': event_time, 'event_type': event_type, 'event_value': event_value, 'patient_id': [pat_id] * len(event_tok)})

        all_data = pd.concat([all_data, data])

        if index == max_sequences:
            break
        else:
            index += 1

    # Make all_data and patients into parquet files
    df_patients.to_parquet(os.path.join(data_path, 'patients.parquet'))
    all_data.to_parquet(os.path.join(data_path, 'data.parquet'))
