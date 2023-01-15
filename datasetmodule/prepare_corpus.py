import math

import pandas as pd
from datetime import timedelta
from tqdm import tqdm
import os

from Utils.Corpus import Corpus
from Utils.Sequence import Sequence
from Utils.Tokenizer import Tokenizer
from Utils.utils import save_corpus


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
    forloeb = forloeb.sample(frac=1)
    diagnosis = pd.read_parquet(os.path.join(data_path, f'{diagnosis_file}.parquet'))
    labtests = pd.read_parquet(os.path.join(data_path, f'{labtest_file}.parquet'))
    vitals = pd.read_parquet(os.path.join(data_path, f'{vitals_file}.parquet'))
    prescs = pd.read_parquet(os.path.join(data_path, f'{pres_file}.parquet'))
    procs = pd.read_parquet(os.path.join(data_path, f'{proc_file}.parquet'))
    adm = pd.read_parquet(os.path.join(data_path, f'{adm_file}.parquet'))

    # Sort on time for late optimization
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
    sequence_list = []

    index = 0
    for _, row in tqdm(forloeb.iterrows(), total=forloeb.shape[0]):

        # Create new forloeb
        pat_id = row['patientid']
        ed_start = row['ankomst_ska']
        ed_end = row['afgang_ska']
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

        mortality_30 = True if row['mortality_30'] == 1 else False
        uns_diag = get_is_unspecific(row['last_diagnosis'])
        req_hosp = True if ed_end != hosp_end else False
        ed_length_of_stay = (ed_end - ed_start).total_seconds() / 60 / 60 / 24
        hosp_length_of_stay = (hosp_end - ed_end).total_seconds() / 60 / 60 / 24
        length_of_stay = (hosp_end - ed_start).total_seconds() / 60 / 60 / 24

        # Prepare data types
        pd_lab = prepare_typed_data(labtests, pat_id, ed_start, hosp_end, ['event_time', 'npucode', 'value', 'reftext'])
        pd_diag = prepare_typed_data(diagnosis, pat_id, ed_start, hosp_end, ['event_time', 'diakod'])
        pd_vitals = prepare_typed_data(vitals, pat_id, ed_start, hosp_end, ['event_time', 'intervention_code', 'value'])
        pd_adm = prepare_typed_data(adm, pat_id, ed_start, hosp_end, ['event_time', 'event_code'])
        pd_procs = prepare_typed_data(procs, pat_id, ed_start, hosp_end, ['event_time', 'event_code'])

        sequence = Sequence(
            pat_id,
            apriori,
            apriori_names,
            pd_lab,
            pd_vitals,
            pd_diag,
            pd_adm,
            pd_procs,
            ed_start,
            ed_end,
            hosp_end,
            ed_length_of_stay,
            hosp_length_of_stay,
            length_of_stay,
            mortality_30,
            req_hosp,
            uns_diag,
            age=demo[1],
            gender=demo[0]
        )

        # Tokenize and create patient string
        sequence.tokenize_all()
        sequence.create_patient_string(['apriori', 'diag', 'lab', 'vital', 'adm', 'proc'])
        sequence.minimize()

        # Check for None values in sequence
        if None in sequence.event_tokens:
            exit(f'None token part of sequence: {sequence.event_tokens}')

        sequence_list.append(sequence)

        if index == max_sequences:
            break
        else:
            index += 1

    print(Tokenizer.not_found)

    print('Creating and saving Corpus')
    corpus = Corpus(data_path, sequence_list)
    corpus.make_corpus_df()
    save_corpus(corpus, os.path.join(data_path, corpus_file))
