from collections import Counter

import torch.nn
from imblearn.over_sampling import SMOTE, RandomOverSampler
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.autograd import Variable

from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
from sklearn import metrics
import shap
import os
import pandas as pd
import torch as th
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from torch.utils.data import DataLoader


from Experiments.Experiment import Experiment
from Models.MortalityModel import MortalityModel
from Models.SingleClassificationModel import SingleClassificationModel
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


def shap_predictions():
    # It wants gradients enabled, and uses the training set
    torch.set_grad_enabled(True)
    e = shap.DeepExplainer(model, Variable(dataset['train_x'][:100]))

    # Get the shap values from my test data (this explainer likes tensors)
    shap_values = e.shap_values(Variable(dataset['test_x']))

    # Plots
    shap.force_plot(explainer.expected_value, shap_values, feature_names)
    #shap.dependence_plot("b1_price_avg", shap_values, data, feature_names)
    shap.summary_plot(shap_values, dataset['test_x'], feature_names)


def perform_oversampling(train_x, train_y):
    # Perform oversampling on train data
    counter = Counter(train_y)
    print("Samples before oversampling")
    print(counter)
    #oversampler = SMOTE()
    oversampler = RandomOverSampler(random_state=0)
    train_x, train_y = oversampler.fit_resample(train_x, train_y)
    counter = Counter(train_y)
    print("Samples after oversampling")
    print(counter)
    return train_x, train_y


def count_measurements(df_samples):
    variables = ['natrium_min', 'natrium_max', 'albumin_min', 'albumin_max', 'haemoglobin_min',
    'haemoglobin_max', 'kalium_min', 'kalium_max', 'calcium_min', 'calcium_max',
    'leukocytter_min', 'leukocytter_max', 'trombocytter_min', 'trombocytter_max',
    'alat_min', 'alat_max', 'crp_min', 'crp_max', 'kreatinin_min', 'kreatinin_max',
    'bilirubin_min', 'bilirubin_max', 'erythrocytter_min', 'erythrocytter_max',
    'karbamid_min', 'karbamid_max', 'nyre-egfr_min', 'nyre-egfr_max',
    'neutrofilocytter_min', 'neutrofilocytter_max', 'rni_temp_min', 'rni_temp_max',
    'rni_respiration_min', 'rni_respiration_max', 'rni_iltsaturation_min', 'rni_iltsaturation_max',
    'rnsr_btsys_min', 'rnsr_btsys_max', 'rnsr_btdia_min', 'rnsr_btdia_max', 'rni_ilttilskud_min', 'rni_ilttilskud_max']

    for var in variables:
        most_occurring = df_samples[var].mode()[0]
        print(f'% occurrences of {var} is {(df_samples[df_samples[var] != most_occurring].shape[0] / df_samples.shape[0]) * 100}')

    exit()

def single_patient_prediction(index, test_data):
        explainerModel = shap.TreeExplainer(model)
        shap_values_Model = explainerModel.shap_values(S)
        p = shap.force_plot(explainerModel.expected_value, shap_values_Model[index], S.iloc[[index]])
        return (p)


def roc_curve_plot(labels, preds):
    fpr, tpr, _ = roc_curve(labels, preds)
    roc_auc = metrics.auc(fpr, tpr)
    print(f'ROC-AUC Score: {roc_auc}')

    plt.figure()
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        label="ROC curve (area = %0.2f)" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic example")
    plt.legend(loc="lower right")
    plt.show()

if __name__ == '__main__':
    data_path = r'\\srvsas9402\Platform_AN\3_Projekt_1\Innovationsprojektet\Data'
    file_name = 'mortality.parquet'
    save_path = r'C:\Users\dunu\PycharmProjects\data\model.pt'
    device = 'cpu'

    train = False
    persist = False
    evaluate = True

    # Load data
    print('Loading samples')
    df_samples = pd.read_parquet(os.path.join(data_path, file_name))
    print(df_samples.shape[0])

    label = ['mortality_30']
    drop_cols = ['last_diagnosis', 'required_hosp', 'hosp_time_remaining'] + label
    data_y = df_samples[label]
    data_x = df_samples.drop(drop_cols, axis=1)

    #count_measurements(data_x)

    # Split data into train and test
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2, stratify=data_y, random_state=0)
    pd_test_x = test_x.copy()
    # Scale data
    train_x, test_x = scale_data(train_x, test_x)

    feature_names = train_x.columns.to_list()
    train_x, train_y = train_x.to_numpy(), train_y.to_numpy().flatten()
    test_x, test_y = test_x.to_numpy(), test_y.to_numpy().flatten()

    # Perform oversampling on train data
    #train_x, train_y = perform_oversampling(train_x, train_y)

    # Convert data to tensor
    train_x = th.tensor(train_x, dtype=th.float32).to(device)
    test_x = th.tensor(test_x, dtype=th.float32).to(device)

    train_y = th.tensor(np.expand_dims(train_y, axis=1), dtype=th.float32).to(device)
    test_y = th.tensor(np.expand_dims(test_y, axis=1), dtype=th.float32).to(device)

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
        h_dim=500,
        o_dim=1,
        n_layers=3,
        dropout=0.1,
        out_type='binary').to(args['device'])

    if train:
        exp = Experiment(args, model, target='binary')
        model = exp.train(train_loader, test_loader)

    if persist:
        save_model(save_path, model)

    if evaluate:

        explainerModel = shap.DeepExplainer(model, Variable(train_x[:1000]))
        shap_values_Model = explainerModel.shap_values(Variable(test_x))
        shap.force_plot(explainerModel.expected_value, shap_values_Model[27], test_x[27], feature_names=feature_names)
        shap.force_plot(explainerModel.expected_value, shap_values_Model[73], test_x[73], feature_names=feature_names)
        shap.force_plot(explainerModel.expected_value, shap_values_Model[80], test_x[80], feature_names=feature_names)
        shap.force_plot(explainerModel.expected_value, shap_values_Model[0], test_x[0], feature_names=feature_names)
        shap.force_plot(explainerModel.expected_value, shap_values_Model[1], test_x[1], feature_names=feature_names)
        shap.force_plot(explainerModel.expected_value, shap_values_Model[2], test_x[2], feature_names=feature_names)
        plt.show()
        exit()

        print("Begin")
        e = shap.DeepExplainer(model, Variable(train_x[:100]))


        print("End")
        # Get the shap values from my test data (this explainer likes tensors)
        shap_values = e.shap_values(Variable(test_x))
        print('End')

        # Plots
        #shap.force_plot(explainer.expected_value, shap_values, feature_names)
        # shap.dependence_plot("b1_price_avg", shap_values, data, feature_names)
        shap.summary_plot(shap_values, test_x, feature_names, plot_type='bar')
        shap.summary_plot(shap_values, test_x, feature_names)

        exit()

        print("Evaluating Model")
        load_model(save_path, model)
        model.eval()
        with torch.no_grad():
            loss_all = 0.0
            all_labs = torch.Tensor()
            all_preds = torch.Tensor()
            for i, (inputs, labels, idx) in enumerate(test_loader, 1):
                with torch.no_grad():
                    preds = model(inputs)

        roc_curve_plot(labels, preds)

