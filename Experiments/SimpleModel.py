import os
from collections import Counter

import pandas as pd
import torch as th
import numpy as np
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier

from Experiment import Experiment
from Models.BaseModel import BaseModel
from Models.SingleClassificationModel import SingleClassificationModel
from TabularDataset import LOSDataset
from utils import save_model, load_model


def scale_data(train_x, test_x, prediction_time):
    # Some columns are good as they are
    if prediction_time == 'fam_end':
        columns_to_scale = ['natrium_min', 'natrium_max', 'albumin_min', 'albumin_max', 'haemoglobin_min',
                            'haemoglobin_max', 'kalium_min', 'kalium_max', 'calcium_min', 'calcium_max',
                            'leukocytter_min', 'leukocytter_max', 'trombocytter_min', 'trombocytter_max',
                            'alat_min', 'alat_max', 'crp_min', 'crp_max', 'kreatinin_min', 'kreatinin_max',
                            'bilirubin_min', 'bilirubin_max', 'erythrocytter_min', 'erythrocytter_max',
                            'karbamid_min', 'karbamid_max', 'nyre-egfr_min', 'nyre-egfr_max',
                            'neutrofilocytter_min', 'neutrofilocytter_max', 'rni_temp_min', 'rni_temp_max',
                            'rni_respiration_min', 'rni_respiration_max', 'rni_iltsaturation_min', 'rni_iltsaturation_max',
                            'rnsr_btsys_min', 'rnsr_btsys_max', 'rnsr_btdia_min', 'rnsr_btdia_max', 'rni_ilttilskud_min', 'rni_ilttilskud_max',
                            'age', 'ccm_score', 'triage_kat', 'hours_since_ska']
    elif prediction_time == 'fam_start':
        columns_to_scale = ['age', 'ccm_score', 'triage_kat']
    print('Scaling Data')
    transformer = QuantileTransformer()
    transformer = transformer.fit(train_x[columns_to_scale])

    # Scale train and test
    train_x[columns_to_scale] = transformer.transform(train_x[columns_to_scale])
    test_x[columns_to_scale] = transformer.transform(test_x[columns_to_scale])

    return train_x, test_x


def perform_oversampling(train_x, train_y):
    # Perform oversampling on train data
    print("Samples before oversampling")
    groups = train_y.columns.to_list()
    num_groups = list(range(1, len(groups) + 1))
    train_y['label'] = 0
    for column, value in zip(groups, num_groups):
        train_y['label'] += train_y[column] * value
    counter = Counter(train_y['label'])
    print(counter)
    #oversampler = SMOTE()
    oversampler = SMOTE(random_state=0)
    train_x, train_y = oversampler.fit_resample(train_x, train_y['label'])
    counter = Counter(train_y)
    print("Samples after oversampling")
    train_y = pd.get_dummies(train_y)
    train_y.columns = groups
    print(counter)

    return train_x, train_y


def scale_label(train_y, test_y):
    columns_to_scale = ['hosp_time_remaining']

    scaler = StandardScaler()
    scaler = scaler.fit(train_y[columns_to_scale])

    train_y[columns_to_scale] = scaler.transform(train_y[columns_to_scale])
    test_y[columns_to_scale] = scaler.transform(test_y[columns_to_scale])

    return train_y, test_y


def remove_outliers(df_samples, days):
    max_hosp_days = days
    df_samples = df_samples[df_samples['hosp_time_remaining'] <= max_hosp_days]

    return df_samples


def binarize_samples(df_samples, days):
    df_samples['hosp_time_remaining'] = np.where(df_samples['hosp_time_remaining'] > days, 1, 0)
    return df_samples


def group_sampels(df_samples, groups):
    group_names = []
    for start, end in zip(groups[0:-1], groups[1:]):
        group = f'hosp_{start}-{end}'
        df_samples[group] = np.where((df_samples['hosp_time_remaining'] > start) & (df_samples['hosp_time_remaining'] <= end), 1, 0)
        group_names.append(group)

    print('Samples in each group')
    for group in group_names:
        print(f'Group {group} has {df_samples[group].sum()} examples')

    return df_samples, group_names


if __name__ == '__main__':

    # ['fam_start', 'fam_end]
    prediction_time = 'fam_end'

    # ['hosp', 'binary', 'groups', 'regression']
    prediction_task = 'binary'

    binary_cutoff = 4.0
    cutoff_days = 30
    groups = [0, 3, 10, cutoff_days]
    outputs = 1

    train = True
    persist = True
    evaluate = True

    data_path = r'\\srvsas9402\Platform_AN\3_Projekt_1\Innovationsprojektet\Data'
    if prediction_time == 'fam_start':
        file_name = 'fam_begin.parquet'
    elif prediction_time == 'fam_end':
        file_name = 'fam_end.parquet'

    save_path = r'C:\Users\dunu\PycharmProjects\data\model.pt'
    device = 'cpu'

    # Load data
    print('Loading samples')
    df_samples = pd.read_parquet(os.path.join(data_path, file_name))

    # Remove outliers
    df_samples = remove_outliers(df_samples, cutoff_days)

    # Filter for hospitalizations
    df_samples = df_samples[df_samples['hosp_time_remaining'] > 0]

    drop_cols, labels = [], []
    if prediction_task == 'groups':
        df_samples, labels = group_sampels(df_samples, groups)
        drop_cols = ['hosp_time_remaining', 'required_hosp'] + labels
        outputs = len(groups) - 1
    if prediction_task == 'binary':
        df_samples = binarize_samples(df_samples, binary_cutoff)
        labels = ['hosp_time_remaining']
        drop_cols = ['required_hosp']
    elif prediction_task == 'hosp':
        labels = ['required_hosp']
        drop_cols = ['hosp_time_remaining']
    elif prediction_task == 'regression':
        labels = ['hosp_time_remaining']
        drop_cols = ['required_hosp']

    drop_cols = drop_cols + labels
    if prediction_time == 'fam_end':
        drop_cols = drop_cols + ['mortality_30', 'last_diagnosis']

    data_y = df_samples[labels]
    data_x = df_samples.drop(drop_cols, axis=1)

    # Split data into train and test
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2, random_state=0)

    # Perform data scaling
    train_x, test_x = scale_data(train_x, test_x, prediction_time)

    # Perform oversampling
    #train_x, train_y = perform_oversampling(train_x, train_y)
    """
    # Sklearn Model
    los_classifier = RandomForestRegressor(max_depth=2, random_state=0)
    los_classifier.fit(train_x, train_y)
    train_preds = los_classifier.predict(train_x)
    test_preds = los_classifier.predict(test_x)
    print(f'Train RFC AUC: {metrics.roc_auc_score(train_y, train_preds)}')
    print(f'Test RFC AUC: {metrics.roc_auc_score(test_y, test_preds)}')

    los_classifier = MLPRegressor(random_state=1, max_iter=300)
    los_classifier.fit(train_x, train_y)
    train_preds = los_classifier.predict(train_x)
    test_preds = los_classifier.predict(test_x)
    print(f'Train MLP AUC: {metrics.roc_auc_score(train_y, train_preds)}')
    print(f'Test MLP AUC: {metrics.roc_auc_score(test_y, test_preds)}')
    """
    # Pytorch Model
    train_x = th.tensor(train_x.to_numpy(), dtype=th.float32).to(device)
    test_x = th.tensor(test_x.to_numpy(), dtype=th.float32).to(device)

    train_y = th.tensor(train_y.to_numpy(), dtype=th.float32).to(device)
    test_y = th.tensor(test_y.to_numpy(), dtype=th.float32).to(device)

    # Add data to dataset
    tabular_train = LOSDataset(train_x, train_y)
    tabular_test = LOSDataset(test_x, test_y)

    # ------------------------- Model Training --------------------------
    args = {'device': 'cpu', 'epochs': 100, 'mini_batch': True, 'batch_size': 256, 'lr': 0.0001, 'weight_decay': 0.01}

    train_loader = DataLoader(tabular_train, batch_size=args['batch_size'], shuffle=False, num_workers=0)
    test_loader = DataLoader(tabular_test, len(tabular_test), shuffle=False, num_workers=0)

    # Setup Model
    model = SingleClassificationModel(
        i_dim=train_x.shape[1],
        h_dim=200,
        o_dim=outputs,
        n_layers=3,
        dropout=0.2,
        out_type=prediction_task).to(args['device'])

    print("Training Pytorch model")
    exp = Experiment(args, model, target=prediction_task)
    model = exp.train(train_loader, test_loader)


"""
    # Extract the found instances using the hosp_classifier
    train_boolean_index = hosp_classifier.predict(train_x).astype(bool)
    test_boolean_index = hosp_classifier.predict(test_x).astype(bool)

    train_x_subset = train_x[train_boolean_index]
    train_y_subset = train_y_all[train_boolean_index]['hosp_time_remaining']
    test_x_subset = test_x[test_boolean_index]
    test_y_subset = test_y_all[test_boolean_index]['hosp_time_remaining']

    print("Training for LOS regression")
    #los_classifier = RandomForestRegressor(max_depth=2, random_state=0)
    los_classifier = MLPRegressor(random_state=1, max_iter=300)

    los_classifier.fit(train_x_subset, train_y_subset)
    train_los_pred = los_classifier.predict(train_x_subset)
    test_los_pred = los_classifier.predict(test_x_subset)
    naive_los_pred = [4.48] * len(train_x_subset)

    print(f'Naive MAE: {metrics.mean_absolute_error(train_y_subset, naive_los_pred)}')
    print(f'Train MAE: {metrics.mean_absolute_error(train_y_subset, train_los_pred)}')
    print(f'Test MAE: {metrics.mean_absolute_error(test_y_subset, test_los_pred)}')
"""