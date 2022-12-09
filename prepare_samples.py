import time

import numpy as np

from utils import load_corpus
from matplotlib import pyplot as plt
import os


def get_outlier_ranges(data, diversion):
    Q3 = np.quantile(data, 0.75)
    Q1 = np.quantile(data, 0.25)
    IQR = Q3 - Q1
    lower_range = Q1 - diversion * IQR
    upper_range = Q3 + diversion * IQR
    return lower_range, upper_range


if __name__ == '__main__':

    # TODO: create some system for managing file paths
    data_path = r'\\srvsas9402\Platform_AN\3_Projekt_1\Innovationsprojektet\Data'
    corpus_file = 'corpus'

    corpus = load_corpus(os.path.join(data_path, corpus_file))

    subset_corpus = corpus.get_subset_corpus(req_hosp=True)

    #subset_corpus.cut_sequences_by_hours(12)

    seq_lengths = subset_corpus.get_sequence_lengths()
    lower_range, upper_range = get_outlier_ranges(seq_lengths, 3)
    outliers = [elm for elm in seq_lengths if elm < lower_range or elm > upper_range]
    inliers = [elm for elm in seq_lengths if elm >= lower_range and elm <= upper_range]

    print(f'Inliers: {len(inliers)}, Outliers: {len(outliers)}')

    n, bins, patches = plt.hist(inliers, 50, density=True, facecolor='g', alpha=0.75)

    # plt.xlabel('Smarts')
    # plt.ylabel('Probability')
    # plt.title('Histogram of IQ')
    # plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
    # plt.xlim(40, 160)
    # plt.ylim(0, 0.03)
    # plt.grid(True)
    plt.show()



    """
    faulthandler.enable()

    data_path = r'\\srvsas9402\Platform_AN\3_Projekt_1\Innovationsprojektet\Data'

    # ['los_buckets', 'los_hourly_test', 'mortality']
    sample_type = 'counts'

    # Persist the forloeb
    if sample_type == 'mortality':
        data_file = 'forloeb_mortality.pkl'
        samples_file = 'mortality.parquet'
    elif sample_type == 'los_buckets':
        data_file = 'forloeb_los.pkl'
        samples_file = 'los_buckets.parquet'
    elif sample_type == 'los_hourly_test':
        data_file = 'forloeb_los.pkl'
        samples_file = 'los_hourly_test.parquet'
    elif sample_type == 'fam_begin':
        data_file = 'forloeb_los.pkl'
        samples_file = 'fam_begin.parquet'
    elif sample_type == 'fam_end':
        data_file = 'forloeb_los.pkl'
        samples_file = 'fam_end.parquet'
    elif sample_type == 'counts':
        data_file = 'forloeb_los.pkl'
        samples_file = 'counts.pickle'

    forloeb = load_data(os.path.join(data_path, data_file))

    all_samples = []
    feat_names = []

    if sample_type == 'counts':
        for i, value in tqdm(enumerate(forloeb), total=len(forloeb)):
            sample = value.get_counts()
            all_samples.append(sample)
            if i == 1000:
                break

        pickle_save(all_samples, os.path.join(data_path, samples_file))

    else:
        for i, value in tqdm(enumerate(forloeb), total=len(forloeb)):
            if sample_type == 'los_buckets':
                samples, feat_names = value.get_samples_by_buckets(1)
            elif sample_type == 'mortality':
                samples, feat_names = value.get_samples_by_buckets(1, end_time='hosp')
            elif sample_type == 'los_hourly_test':
                test_begin = datetime.datetime(2021, 1, 1)
                samples, feat_names = value.get_samples_hourly_split(3, test_begin)
            elif sample_type == 'fam_begin':
                samples, feat_names = value.get_sample_fam_begin()
            elif sample_type == 'fam_end':
                samples, feat_names = value.get_sample_fam_end()

            all_samples.extend(samples)

        # Create pandas dataframe from the samples
        all_samples = pd.DataFrame(all_samples, columns=feat_names)

        # Persist the dataframe as parquet
        all_samples.to_parquet(os.path.join(data_path, samples_file))
    """
