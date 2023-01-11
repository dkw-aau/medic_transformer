import os
from bisect import bisect

import matplotlib.pyplot as plt
import numpy as np
from Utils.utils import load_corpus
from config import Config
import matplotlib.dates as mdates


def length_og_stay_plot(corpus, bins, cutoff_days, file_name='los_bins'):
    fig, axs = plt.subplots(1, 1)
    in_liers = [seq.length_of_stay for seq in corpus.sequences if 0 < seq.length_of_stay < cutoff_days]
    out_liers = [seq.length_of_stay for seq in corpus.sequences if seq.length_of_stay >= cutoff_days]
    zero_time = [seq.length_of_stay for seq in corpus.sequences if seq.length_of_stay == 0]

    print(f'With a cutoff of {cutoff_days} days we get '
          f'{len(in_liers)} samples and {len(out_liers)} outliers')
    print(f'{len(zero_time)} patients did not have an admission, these are omitted from the plot\n')

    axs.hist(in_liers, bins=bins)
    axs.set_title(f'LOS for {len(in_liers)} patient sequences')
    axs.set_ylabel('Num Sequences')
    axs.set_xlabel('LOS Days')
    plt.savefig(f'{args.path["out_fold"]}/{file_name}.png')


def length_og_stay_category_plot(corpus, categories, file_name='los_cat'):
    bins = {}
    inliers = [bisect(categories, seq.length_of_stay) for seq in corpus.sequences]
    for i in range(0, len(categories) + 1):
        bins[i] = inliers.count(i)

    fig, axs = plt.subplots()
    axs.bar([str(x) for x in ['0'] + categories], bins.values())

    axs.set_title(f'Label distribution for {len(inliers)} patient sequences')
    axs.set_ylabel('Num Sequences')
    axs.set_xlabel('Classes')

    plt.savefig(f'{args.path["out_fold"]}/{file_name}.png')


def hist_seq_len(corpus, bins, cutoff_len, file_name='seq_len'):
    inliers = [len(seq.event_tokens) for seq in corpus.sequences if len(seq.event_tokens) < cutoff_len]
    outliers = [len(seq.event_tokens) for seq in corpus.sequences if len(seq.event_tokens) >= cutoff_len]

    print('--- Sequence Lengths ---')
    print(f'Outliers are specified as sequences with more than {cutoff_len} tokens')
    print(f'Smallest and largest sequences: {min(inliers)} - {max(inliers)}')
    print(f'{len(inliers)} inliers')
    print(f'{len(outliers)} outliers')
    print(f'Mean sequence length of inliers: {sum(inliers) / len(inliers)}')

    fig, axs = plt.subplots(1, 1)
    axs.hist(inliers, bins=bins)
    axs.set_title(f'Sequence length of {len(inliers)} inliers')
    axs.set_ylabel('Num Sequences')
    axs.set_xlabel('Sequence Length')
    plt.savefig(f'{args.path["out_fold"]}/{file_name}.png')


def token_distribution(corpus, bins, cutoff_min, cutoff_max, file_name='token_dist'):
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
    axs.set_title(f'Distribution of tokens for {len(dist.keys())} distinct inlier tokens')
    axs.set_ylabel('Num Tokens')
    axs.set_xlabel('Num Occurrences')
    plt.savefig(f'{args.path["out_fold"]}/{file_name}.png')


def apriori_distribution(corpus):
    pass


def token_type_time_dist(corpus, file_name='tok_typ_dist'):
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
    plt.savefig(f'{args.path["out_fold"]}/{file_name}.png')


def type_distribution(corpus, file_name=f'event_type_dist'):
    event_types = ['diag', 'lab', 'vital', 'apriori']
    event_type_counts = {event: 0 for event in event_types}
    for sequence in corpus.sequences:
        for event_type in event_types:
            event_type_counts[event_type] += sequence.event_types.count(event_type)

    counts = [event_count / len(corpus.sequences) for event_count in event_type_counts.values()]
    fig, axs = plt.subplots()
    axs.bar(event_types, counts)

    axs.set_title('Token Type Distribution')
    axs.set_ylabel(f'Avg samples for different event types')
    axs.set_xlabel('Token Types')

    plt.savefig(f'{args.path["out_fold"]}/{file_name}.png')


def position_distribution(corpus, file_name='pos_dist'):
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

    plt.savefig(f'{args.path["out_fold"]}/{file_name}.png')


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

    # Load Corpus and subset corpus
    corpus = load_corpus(os.path.join(args.path['data_fold'], args.corpus_name))
    print(len(corpus.sequences))
    # Select subset corpus
    min_hours = 24
    cut_corpus = corpus.get_subset_by_min_hours(min_hours=min_hours)
    cut_corpus.cut_sequences_by_hours(hours=min_hours)
    cut_corpus.substract_los_hours(hours=min_hours)

    # Length of stay
    length_og_stay_plot(corpus, bins=50, cutoff_days=14)
    length_og_stay_category_plot(corpus, categories=[1, 2, 3, 4, 5, 6, 7, 14])

    # Length of stay
    length_og_stay_plot(cut_corpus, bins=50, cutoff_days=14, file_name=f'los_bins_{min_hours}')
    length_og_stay_category_plot(cut_corpus, categories=[1, 2, 3, 4, 5, 6, 7, 14], file_name=f'los_cat_{min_hours}')

    # Sequence lengths
    hist_seq_len(corpus, bins=50, cutoff_len=256, file_name='seq_len')
    hist_seq_len(cut_corpus, bins=50, cutoff_len=256, file_name=f'seq_len_{min_hours}')

    # Token distribution
    token_distribution(corpus, bins=200, cutoff_min=0, cutoff_max=20000, file_name='token_dist_all')
    token_distribution(cut_corpus, bins=200, cutoff_min=10, cutoff_max=20000, file_name='token_dist_min_10')

    # TODO: Investigate Apriori
    # Are there any apriori concepts that are not seen 200 times?
    apriori_distribution(corpus)

    # Token type distribution over time
    token_type_time_dist(corpus, file_name='tok_typ_dist')
    token_type_time_dist(cut_corpus, file_name=f'tok_typ_dist_{min_hours}')

    # Plot a sequence
    # plot_sequence(corpus, 3)

    # Token type distribution plot
    type_distribution(corpus, file_name='event_type_dist')
    type_distribution(cut_corpus, file_name=f'event_type_dist_{min_hours}')

    position_distribution(corpus, file_name='pos_dist')
    position_distribution(cut_corpus, file_name=f'pos_dist_{min_hours}')
    exit()

    # TODO: Analyze the Aleatoric uncertainty of the problem by finding similar looking sequences
    # TODO: and look at the distribution in their LOS