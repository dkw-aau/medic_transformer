import os
import time

from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from Utils.utils import load_corpus
from config import Config
import matplotlib.dates as mdates
import statistics


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


def hist_los(corpus, bins, cutoff_days):
    fig, axs = plt.subplots(1, 1)
    in_liers = [seq.length_of_stay for seq in corpus.sequences if 0 < seq.length_of_stay < cutoff_days]
    out_liers = [seq.length_of_stay for seq in corpus.sequences if seq.length_of_stay >= cutoff_days]
    zero_time = [seq.length_of_stay for seq in corpus.sequences if seq.length_of_stay == 0]

    print(f'With a cutoff of {cutoff_days} days we get '
          f'{len(in_liers)} samples and {len(out_liers)} outliers')
    print(f'{len(zero_time)} patients did not have an admission, these are omitted from the plot\n')

    axs.hist(in_liers, bins=bins)
    plt.savefig(f'{args.path["out_fold"]}/hist_los.png')


def hist_los_adm(corpus, bins, cutoff_hours_min, cutoff_hours_max):
    outliers_less = [seq for seq in corpus.sequences if seq.length_of_stay * 24 < cutoff_hours_min]
    outliers_more = [seq for seq in corpus.sequences if seq.length_of_stay * 24 > cutoff_hours_max]
    inliers = [seq for seq in corpus.sequences if cutoff_hours_min < seq.length_of_stay * 24 < cutoff_hours_max]

    print(f'--- LOS more than {cutoff_hours_min} hours admissions ---')
    print(f'Outliers with less than {cutoff_hours_min} hours of admission: {len(outliers_less)}')
    print(f'Outliers with more than {cutoff_hours_max} hours of admission: {len(outliers_more)}')
    print(f'Remaining patients: {len(inliers)} ~ {round(len(inliers) / len(corpus.sequences), 2)}% of of the data')
    print(f'Mean los: {sum([seq.length_of_stay for seq in inliers]) / len(inliers)} days')
    print(f'Median los: {statistics.median([seq.length_of_stay for seq in inliers])} days')
    print(f'No hospitalization in {len([seq for seq in inliers if seq.req_hosp is False])} cases\n')

    fig, axs = plt.subplots(1, 1)
    axs.hist([seq.length_of_stay for seq in inliers], bins=bins)
    axs.set_title(f'Sequence length of {len(inliers)} patient admissions')
    plt.savefig(f'{args.path["out_fold"]}/los_adm.png')


def hist_seq_len(corpus, bins, cutoff_len):
    inliers = [len(seq.event_tokens) for seq in corpus.sequences if len(seq.event_tokens) < cutoff_len]
    outliers = [len(seq.event_tokens) for seq in corpus.sequences if len(seq.event_tokens) >= cutoff_len]
    apriori = [len(seq.event_tokens) for seq in corpus.sequences if len(seq.event_tokens) == seq.apriori_len]

    print('--- Sequence Lengths ---')
    print(f'Outliers are specified as sequences with more than {cutoff_len} tokens')
    print(f'Smallest and largest sequences: {min(inliers)} - {max(inliers)}')
    print(f'{len(inliers)} inliers')
    print(f'{len(outliers)} outliers')
    print(f'{len(apriori)} with only apriori knowledge')
    print(f'Mean sequence length of inliers: {sum(inliers) / len(inliers)}')
    print(f'Mean sequence length of outliers: {sum(outliers) / len(outliers)}\n')

    fig, axs = plt.subplots(1, 1)
    axs.hist(inliers, bins=bins)
    axs.set_title(f'Sequence length of {len(inliers)} inliers')
    plt.savefig(f'{args.path["out_fold"]}/hist_tokens.png')


def token_distribution(corpus, bins, cutoff_min, cutoff_max):
    dist = {}
    for seq in corpus.sequences:
        for token in seq.event_tokens:
            dist[token] = dist[token] + 1 if token in dist else 1

    print('--- Token Distribution ---')

    # Remove outliers
    dist_below = {k: v for k, v in dist.items() if v < cutoff_min}
    dist_above = {k: v for k, v in dist.items() if v > cutoff_max}
    dist_inlier = {k: v for k, v in dist.items() if cutoff_min < v < cutoff_max}

    print(f'{len(dist_below.keys())} concepts with fewer than {cutoff_min} occurrences accounting for {sum(dist_below.values())} tokens')
    print(f'{len(dist_above.keys())} concepts with more than {cutoff_max} occurrences accounting for {sum(dist_above.values())} tokens')
    print(f'{len(dist_inlier.keys())} concepts within the bounds accounting for {sum(dist_inlier.values())} tokens\n')

    # Token distribution
    fig, axs = plt.subplots(1, 1)
    axs.hist(dist_inlier.values(), bins=bins)
    axs.set_title(f'Token distributions for {len(dist.keys())} distinct inlier tokens')
    plt.savefig(f'{args.path["out_fold"]}/hist_token_distribution.png')


def apriori_distribution(corpus):
    pass


def time_hist(corpus):
    year_dict = {}
    year_type_dict = {}

    for seq in corpus.sequences:
        year = seq.ed_start.year
        if year in year_dict:
            year_dict[year].append(seq)
        else:
            year_dict[year] = [seq]

    sorted_year_dict = dict(sorted(year_dict.items()))

    event_types = ['diag', 'lab', 'vital', 'apriori']
    for year, sequences in year_dict.items():
        year_type_dict[year] = {}
        for type in event_types:
            for seq in sequences:
                if type in year_type_dict[year]:
                    year_type_dict[year][type] += seq.event_types.count(type)
                else:
                    year_type_dict[year][type] = seq.event_types.count(type)

    sorted_year_types = dict(sorted(year_type_dict.items()))

    data = np.empty([len(sorted_year_types), len(event_types)])
    for i, (year, values) in enumerate(sorted_year_types.items()):
        for k, val in enumerate(values.values()):
            data[i][k] = val / len(sorted_year_dict[year])

    labels = event_types

    x = np.arange(len(labels))  # the label locations
    width = 0.1  # the width of the bars

    fig, ax = plt.subplots()
    rects = [ax.bar(x + width * i * 4 / len(labels), data[i], width, label=list(sorted_year_types.keys())[i]) for i in range(0, len(data))]

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('Scores by group and gender')
    ax.set_xticks(x, labels)
    ax.legend()

    for rect in rects:
        ax.bar_label(rect, padding=3)

    fig.tight_layout()
    plt.savefig(f'{args.path["out_fold"]}/token_type_distribution.png')


def type_distribution(corpus):
    event_types = ['diag', 'lab', 'vital', 'apriori']
    event_type_counts = {event: 0 for event in event_types}
    for sequence in corpus.sequences:
        for event_type in event_types:
            event_type_counts[event_type] += sequence.event_types.count(event_type)

    counts = [event_count / len(corpus.sequences) for event_count in event_type_counts.values()]
    fig, ax = plt.subplots()
    ax.bar(event_types, counts)

    plt.bar(event_types, counts, align='center', alpha=0.5)
    plt.ylabel(f'Avg samples for different event types')
    plt.xticks(rotation=45)

    plt.savefig(f'{args.path["out_fold"]}/event_type_distribution.png')


def position_distribution(corpus):
    event_types = ['diag', 'lab', 'vital', 'apriori']
    event_type_counts = {event: 0 for event in event_types}
    for sequence in corpus.sequences:
        tmp_event_pos = -1
        for i, event_pos in enumerate(sequence.event_pos_ids):
            if tmp_event_pos != event_pos:
                event_type_counts[sequence.event_types[i]] += 1
                tmp_event_pos = event_pos

    counts = [event_count / len(corpus.sequences) for event_count in event_type_counts.values()]
    fig, ax = plt.subplots()
    ax.bar(event_types, counts)

    plt.bar(event_types, counts, align='center', alpha=0.5)
    plt.ylabel(f'Avg positions for different event types')
    plt.xticks(rotation=45)

    plt.savefig(f'{args.path["out_fold"]}/event_pos_distribution.png')


def plot_sequence(corpus, seq_num):
    names = corpus.sequences[seq_num].event_tokens[41:]
    dates = corpus.sequences[seq_num].event_times[41:]

    # Choose some nice levels
    levels = np.tile([-5, 5, -3, 3, -1, 1],
                     int(np.ceil(len(dates) / 6)))[:len(dates)]

    # Create figure and plot a stem plot with the date
    fig, ax = plt.subplots(figsize=(8.8, 4), constrained_layout=True)
    ax.set(title='Patient sequence ecents')

    markerline, stemline, baseline = ax.stem(dates, levels,
                                             linefmt="C3-", basefmt="k-",
                                             use_line_collection=True)

    plt.setp(markerline, mec="k", mfc="w", zorder=3)

    # Shift the markers to the baseline by replacing the y-data by zeros.
    markerline.set_ydata(np.zeros(len(dates)))

    # annotate lines
    vert = np.array(['top', 'bottom'])[(levels > 0).astype(int)]
    for d, l, r, va in zip(dates, levels, names, vert):
        ax.annotate(r, xy=(d, l), xytext=(-3, np.sign(l) * 3),
                    textcoords="offset points", va=va, ha="right")

    # format xaxis with 1 hour intervals
    ax.get_xaxis().set_major_locator(mdates.HourLocator(interval=4))
    ax.get_xaxis().set_major_formatter(mdates.DateFormatter("%a %d - %H:%M"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    # remove y axis and spines
    ax.get_yaxis().set_visible(False)
    for spine in ["left", "top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.margins(y=0.2)

    plt.savefig(f'{args.path["out_fold"]}/sequence_plot.png')


# Primary pipeline for analyzing the corpus
if __name__ == '__main__':
    config_file = ['config.ini']
    args = Config(
        file_path=config_file
    )
    start = time.time()
    corpus = load_corpus(os.path.join(args.path['data_fold'], args.corpus_name))

    # Length of stay
    hist_los(corpus, bins=50, cutoff_days=14)

    # LOS for > x hours admission
    hist_los_adm(corpus, bins=50, cutoff_hours_min=24, cutoff_hours_max=720)

    # Sequence lengths
    hist_seq_len(corpus, bins=50, cutoff_len=256)

    # Token distribution
    token_distribution(corpus, bins=50, cutoff_min=200, cutoff_max=1000000)

    # TODO: Investigate Apriori
    # Are there any apriori concepts that are not seen 200 times?
    apriori_distribution(corpus)

    time_hist(corpus)

    plot_sequence(corpus, 43)
    # corpus.sequences[9].event_tokens[40:]

    # Analyze the token distribution for the first x hours of admissions,
    # What does this look like?
    # Cut the corpus by a set amount of hours and dio further investigations
    corpus = corpus.get_subset_by_min_hours(min_hours=24)
    corpus.cut_sequences_by_hours(hours=24)

    type_distribution(corpus)
    position_distribution(corpus)

    # TODO: Analyze the Aleatoric uncertainty of the problem by finding similar looking sequences
    # TODO: and look at the distribution in their LOS