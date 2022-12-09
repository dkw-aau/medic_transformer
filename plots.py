import os

from tqdm import tqdm
from utils import load_data, pickle_save
from utils import vital_labels, lab_labels, vital_names, lab_names
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import json


def count_label_occurrences(data, labels, code_name, patient_wise=False):
    print('Counting labels')
    label_counts = dict()
    for observations in tqdm(data, total=len(data)):
        if not observations.empty:
            observation_groups = observations.groupby(by=code_name)
            for label in labels:
                if label in observation_groups.groups.keys():
                    if label in label_counts:
                        if patient_wise:
                            label_counts[label] += 1
                        else:
                            label_counts[label] += observation_groups.get_group(label).shape[0]
                    else:
                        if patient_wise:
                            label_counts[label] = 1
                        else:
                            label_counts[label] = observation_groups.get_group(label).shape[0]

    return label_counts


def aggregate_label_occurrences(data, labels, code_name, filename, use_preprocessed=True):
    print('Counting labels')
    label_measurements = dict()
    if use_preprocessed:
        label_measurements = load_data(os.path.join(data_path, filename))
    else:
        for observations in tqdm(data, total=len(data)):
            if not observations.empty:
                observation_groups = observations.groupby(by=code_name)
                for label in labels:
                    if label in observation_groups.groups.keys():
                        observation = observation_groups.get_group(label)
                        values = observation['value'].tolist()
                        if label in label_measurements:
                            label_measurements[label].extend(values)
                        else:
                            label_measurements[label] = values

        pickle_save(label_measurements, os.path.join(data_path, filename))

    return label_measurements


def get_outlier_ranges(data, diversion):
    Q3 = np.quantile(data, 0.75)
    Q1 = np.quantile(data, 0.25)
    IQR = Q3 - Q1
    lower_range = Q1 - diversion * IQR
    upper_range = Q3 + diversion * IQR
    return lower_range, upper_range


def get_plot_sizes(num_plots, max_vert, max_hori):
    plot_dims = []
    while num_plots > max_vert * max_hori:
        plot_dims.append([max_vert, max_hori])
        num_plots -= max_vert * max_hori
    if num_plots > 0:
        last_vert = math.ceil(num_plots / max_hori)
        plot_dims.append([last_vert, max_hori])
    return plot_dims


def plot_measurements(measurements, labels, names):
    max_horizontal = 2
    max_vertical = 2
    plot_sizes = get_plot_sizes(len(measurements), max_vertical, max_horizontal)

    plot_counter = 0
    for num_vert, num_hori in plot_sizes:
        fig, axs = plt.subplots(num_vert, num_hori)
        fig.suptitle('Measurements for Patients in the Emergency Department')
        for vert in range(0, num_vert):
            for hori in range(0, num_hori):
                plot_name = names[plot_counter]
                measure_type = labels[plot_counter]
                plot_data = measurements[measure_type]
                num_data = len(plot_data)
                measures_df = pd.Series(plot_data, name=plot_name)
                lower_range, upper_range = get_outlier_ranges(measures_df, 3)
                measures_df = measures_df[(measures_df > lower_range) & (measures_df < upper_range)]
                num_outliers = num_data - len(measures_df)
                mean_value = round(measures_df.mean(), 2)
                median_value = round(measures_df.median(), 2)
                if num_vert > 1:
                    coords = (vert, hori)
                else:
                    coords = hori
                axs[coords].hist(measures_df, 30, density=True, facecolor='g', alpha=0.75)
                axs[coords].set_title(plot_name)
                axs[coords].text(
                    x=0.05,
                    y=0.8,
                    s=f'Observations: {num_data}\n'
                      f'Outliers: {num_outliers}\n'
                      f'Mean Value: {mean_value}\n'
                      f'Median Value: {median_value}',
                    horizontalalignment='left',
                    verticalalignment='center',
                    transform=axs[coords].transAxes)
                plot_counter += 1
                if plot_counter >= len(measurements):
                    break
        fig.set_size_inches(18.5, 10.5)
        plt.show()


def count_vital_measurements(vitals):
    count_label_occurrences(vitals, vital_labels, 'intervention_code')


def print_counts(counts, num_samples, names, plot=False):
    y_values = []
    for (key, value), label in zip(counts.items(), names):
        observed_amount = value / num_samples * 100
        missing_amount = 100 - observed_amount
        print(f'{label} has {value} occurrences amounting to {observed_amount}% of samples')
        y_values.append(missing_amount)

    if plot:
        plt.bar(names, y_values, align='center', alpha=0.5)
        plt.ylabel('% Samples with missing data')
        plt.xticks(rotation=45)
        plt.show()


def get_json_vital(vitals):
    data = {}
    data['key'] = 'value'
    for vital in vitals.iterrows():
        json_vital = {}
        json_vital

    {"timestamp": "6727654800", "category": "Vitals", "chart_time": "2183-03-11 09:00:00 UTC",
     "key": "Diastolic blood pressure", "value_num": 53},

def generate_json(lab_ed, lab_hosp, vitals_ed, vitals_hosp):
    for patient in zip(lab_ed, lab_hosp, vitals_ed, vitals_hosp):
        pass


if __name__ == '__main__':
    data_path = r'\\srvsas9402\Platform_AN\3_Projekt_1\Innovationsprojektet\Data'
    file_name = 'counts_small.pickle'

    # ['ED', 'HOSP', 'BOTH']
    sample_type = 'ED'

    forloeb = load_data(os.path.join(data_path, file_name))

    # Split forloeb into groups of measurements
    print('Splitting data')
    lab_ed, lab_hosp, vitals_ed, vitals_hosp, presc = [], [], [], [], []
    for l_ed, l_hosp, v_ed, v_hosp, p, patient_type in tqdm(forloeb, total=len(forloeb)):
        lab_ed.append(l_ed)
        lab_hosp.append(l_hosp)
        vitals_ed.append(v_ed)
        vitals_hosp.append(v_hosp)
        presc.append(p)

    generate_json(lab_ed, lab_hosp, vitals_ed, vitals_hosp)

    {"timestamp": "6727720800", "category": "Labs", "chart_time": "2183-03-12 03:20:00 UTC", "key": "Potassium",
     "value_num": 4.2999999999999998},

    {"timestamp": "6727654800", "category": "Vitals", "chart_time": "2183-03-11 09:00:00 UTC",
     "key": "Diastolic blood pressure", "value_num": 53},

    {"timestamp": "6727688580", "note": "Test note.\n\nNo real data.", "category": "Nursing/other",
     "charttime": "2183-03-11 18:23:00 UTC"},

    # Also taking into account the patient, find occurences of measurements
    lab_counts = count_label_occurrences(lab_tests, lab_labels, 'npucode', patient_wise=True)
    print_counts(lab_counts, len(forloeb), lab_names, plot=True)
    vital_counts = count_label_occurrences(vitals, vital_labels, 'intervention_code', patient_wise=True)
    print_counts(vital_counts, len(forloeb), vital_names, plot=True)
    exit()

    # Aggregate measurements for each type of data and save for later processing
    vital_measurements = aggregate_label_occurrences(vitals, vital_labels, 'intervention_code', 'vitals.pkl')
    lab_measurements = aggregate_label_occurrences(lab_tests, lab_labels, 'npucode', 'lab_tests.pkl')

    # Plot the measurements
    plot_measurements(vital_measurements, vital_labels, vital_names)
    #plot_measurements(lab_measurements, lab_labels, lab_names)



