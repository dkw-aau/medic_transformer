import os
import pandas as pd
import torch as th
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from Experiment import Experiment
from Models.BaseModel import BaseModel
from TabularDataset import LOSDataset
from utils import save_model, load_model


def scale_data(train_x, test_x):
    # Some columns are good as they are
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

    print('Scaling Data')
    transformer = QuantileTransformer()
    transformer = transformer.fit(train_x[columns_to_scale])

    # Scale train and test
    train_x[columns_to_scale] = transformer.transform(train_x[columns_to_scale])
    test_x[columns_to_scale] = transformer.transform(test_x[columns_to_scale])

    return train_x, test_x


def scale_label(train_y, test_y):
    columns_to_scale = ['hosp_time_remaining']

    scaler = StandardScaler()
    scaler = scaler.fit(train_y[columns_to_scale])

    train_y[columns_to_scale] = scaler.transform(train_y[columns_to_scale])
    test_y[columns_to_scale] = scaler.transform(test_y[columns_to_scale])

    return train_y, test_y


def remove_outliers(df_samples):
    max_hosp_days = 20
    df_samples = df_samples[df_samples['hosp_time_remaining'] < max_hosp_days]

    return df_samples


if __name__ == '__main__':
    data_path = r'\\srvsas9402\Platform_AN\3_Projekt_1\Innovationsprojektet\Data'
    file_name = 'akm_samples.parquet'
    save_path = r'C:\Users\dunu\PycharmProjects\data\model.pt'
    device = 'cpu'

    train = True
    persist = True
    evaluate = True

    # ['LOS', 'LongShort']
    task = 'LOS'

    # Load data
    print('Loading samples')
    df_samples = pd.read_parquet(os.path.join(data_path, file_name))

    # Remove outliers
    df_samples = remove_outliers(df_samples)

    labels = ['required_hosp', 'hosp_time_remaining']
    drop_cols = ['num_vitals', 'num_lab_tests', 'num_presc_cats', 'last_diagnosis'] + labels
    data_y = df_samples[labels]
    data_x = df_samples.drop(drop_cols, axis=1)

    # If task is LongSHort, change data_y accordingly
    if task == 'LongShort':
        data_y['hosp_time_remaining'] = np.where(data_y['hosp_time_remaining'] >= 2, 1, 0)

    # Split data into train and test
    # TODO: Right now train_test split is random, we might want all but the last year for train for some tasks
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2, random_state=0)

    # Plot histogram over data
    train_y[train_y['hosp_time_remaining'] > 0]['hosp_time_remaining'].hist(bins=100)

    # Scale data
    train_x, test_x = scale_data(train_x, test_x)
    #if task == 'LOS':
    #    train_y, test_y = scale_label(train_y, test_y)
    #    train_y[train_y['hosp_time_remaining'] > 0]['hosp_time_remaining'].hist(bins=100)


    # Convert data to tensor
    train_x = th.tensor(train_x.to_numpy(), dtype=th.float32).to(device)
    test_x = th.tensor(test_x.to_numpy(), dtype=th.float32).to(device)

    train_y = th.tensor(train_y.to_numpy(), dtype=th.float32).to(device)
    test_y = th.tensor(test_y.to_numpy(), dtype=th.float32).to(device)

    # Add data to dataset
    tabular_train = LOSDataset(train_x, train_y)
    tabular_test = LOSDataset(test_x, test_y)

    # ------------------------- Model Training --------------------------
    args = {'device': 'cpu', 'epochs': 100, 'mini_batch': True, 'batch_size': 256, 'lr': 0.0001, 'weight_decay': 0}

    train_loader = DataLoader(tabular_train, batch_size=args['batch_size'], shuffle=False, num_workers=0)
    test_loader = DataLoader(tabular_test, len(tabular_test), shuffle=False, num_workers=0)

    # Setup Model
    model = BaseModel(
        i_dim=train_x.shape[1],
        h_dim=25,
        o_dim=1,
        n_layers=3,
        dropout=0.1).to(args['device'])

    if train:
        exp = Experiment(args, model)
        model = exp.train(train_loader, test_loader)

    if persist:
        save_model(save_path, model)

    if evaluate:
        load_model(save_path, model)
