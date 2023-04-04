import datetime
from bisect import bisect

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.pyplot import figure

from Utils.Corpus import Corpus
from config import Config


def time_names():
    data = [(1588540440, 'A06A'), (1588545949, 'ilt_low'), (1588545949, 'respiration_normal'), (1588545949, 'temp_normal'), (1588545949, 'puls_normal'), (1588554900, 'ZZ7098'), (1588558300, 'temp_normal'), (1588558300, 'ilt_low'), (1588564920, 'respiration_normal'), (1588564920, 'temp_normal'), (1588573080, 'NPU02319_low'), (1588573080, 'NPU02840_normal'), (1588575141, 'NPU01370_low'), (1588588601, 'NPU03230_low'), (1588595400, 'respiration_normal'), (1588595400, 'puls_normal'), (1588595400, 'ilttilskud_normal'), (1588595400, 'ilt_low'), (1588595400, 'temp_normal')]
    new_times = []
    for x in [x[0] for x in data]:
        new_times.append(datetime.datetime(1970, 1, 1) + datetime.timedelta(seconds=x))
    new_names = [y for x, y in data]
    print(data)

    return new_times, new_names


def length_og_stay_plot(corpus, cutoff_days, file_name='los_bins'):
    losses = [seq.los for seq in corpus.sequences]
    #out_liers = [seq.los for seq in corpus.sequences if seq.los >= cutoff_days]
    #zero_time = [seq.los for seq in corpus.sequences if seq.los == 0]


    #print(f'With a cutoff of {cutoff_days} days we get'
    #      f'{len(in_liers)} samples and {len(out_liers)} outliers')
    #print(f'{len(zero_time)} patients did not have an admission, these are omitted from the plot\n')

    sns.set_theme()

    fig, ax = plt.subplots()
    bins = np.arange(0, 31, 1)
    _, bins, patches = plt.hist(np.clip(losses, bins[0], bins[-1]), density=True, bins=bins)

    xlabels = [str(int(x)) for x in bins[2::2]]
    xlabels[-1] += '+'

    N_labels = len(xlabels)
    plt.xlim([0, 31])
    x_locations = np.arange(N_labels) * 2 + 1.5
    x_locations[-1] += 0.5
    plt.xticks(x_locations)
    ax.set_xticklabels(xlabels)

    ax.set_title(f'Hospital LOS', fontsize=25)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_ylabel('Density', fontsize=22)
    ax.set_xlabel('Days', fontsize=22)
    fig.subplots_adjust(left=0.15, bottom=0.15)
    plt.setp(patches, linewidth=0)
    fig.tight_layout()
    fig.set_size_inches(10, 6, forward=True)

    plt.savefig(f'{args.path["out_fold"]}/{file_name}.eps', format='eps')


def length_og_stay_category_plot(corpus, label_names, file_name='los_cat'):
    bins = {}
    categories = [2, 7, 14, 30]
    inliers = [bisect(categories, seq.los) for seq in corpus.sequences]
    for i in range(0, len(categories) + 1):
        bins[i] = inliers.count(i)

    sum_bins = sum(bins.values())
    fig, axs = plt.subplots()
    axs.bar(label_names, [x / sum_bins for x in bins.values()])

    sns.set_theme()

    axs.set_title(f'Categorical LOS', fontsize=25)
    axs.set_ylabel('Density', fontsize=22)
    axs.set_xlabel('Days', fontsize=22)
    axs.tick_params(axis='both', which='major', labelsize=16)
    fig.subplots_adjust(left=0.15, bottom=0.15)
    fig.set_size_inches(10, 6, forward=True)
    plt.savefig(f'{args.path["out_fold"]}/{file_name}.eps', format='eps')


def hist_seq_len(corpus, bins, cutoff_len, file_name='seq_len'):
    inliers = [len(seq.event_tokens) for seq in corpus.sequences if len(seq.event_tokens) < cutoff_len]
    outliers = [len(seq.event_tokens) for seq in corpus.sequences if len(seq.event_tokens) >= cutoff_len]

    print('--- Sequence Lengths ---')
    print(f'Outliers are specified as sequences with more than {cutoff_len} tokens')
    print(f'Smallest and largest sequences: {min(inliers)} - {max(inliers)}')
    print(f'{len(inliers)} inliers')
    print(f'{len(outliers)} outliers')
    print(f'Mean sequence length of inliers: {sum(inliers) / len(inliers)}\n')

    sns.set_theme()

    fig, axs = plt.subplots(1, 1)
    axs.hist(inliers, bins=bins)
    axs.set_title(f'Length of Patient Sequences', fontsize=20)
    axs.set_ylabel('Number of Sequences', fontsize=16)
    axs.set_xlabel('Number of Token Events', fontsize=16)
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


def token_type_time_dist(corpus, file_name='tok_typ_dist', event_types=None):
    year_dict = {}
    year_type_dict = {}

    for seq in corpus.sequences:
        year = seq.ed_start.year
        if year in year_dict:
            year_dict[year].append(seq)
        else:
            year_dict[year] = [seq]

    sorted_year_dict = dict(sorted(year_dict.items()))

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


def type_distribution(corpus, file_name=f'event_type_dist', event_types=None):
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


def distinct_token_types(corpus, file_name=f'distinct_token_types', event_types=None):
    event_types_distinct = {event: 0 for event in event_types}
    event_type_counts = {event: 0 for event in event_types}
    event_tokens_distinct = {event: [] for event in event_types}

    for event in event_types:
        token_set = set()
        for seq in corpus.sequences:
            event_type_tokens = [t_val for t_val, t_type in zip(seq.event_tokens, seq.event_types) if t_type == event]
            token_set.update(set(event_type_tokens))
            event_type_counts[event] += len(event_type_tokens)
        event_tokens_distinct[event] = token_set
        event_types_distinct[event] = len(token_set)

    fig, axs = plt.subplots()
    axs.bar(event_types, event_types_distinct.values())

    print('Token types and their distinct token counts')
    print(f'{event_types_distinct}')
    print('Token types and their total token counts')
    print(f'{event_type_counts}\n')
    #for key, value in event_tokens_distinct.items():
    #    print(key, value)

    axs.set_title('Token Types and their Distinct Concepts')
    axs.set_ylabel(f'Count Distinct Concepts')
    axs.set_xlabel('Token Types')

    plt.savefig(f'{args.path["out_fold"]}/{file_name}.png')


def position_distribution(corpus, file_name='pos_dist', event_types=None):
    event_type_counts = {event: 0 for event in event_types}
    for seq in corpus.sequences:
        tmp_event_pos = -1
        for i, event_pos in enumerate(seq.event_pos_ids):
            if tmp_event_pos != event_pos:
                event_type_counts[seq.event_types[i]] += 1
                tmp_event_pos = event_pos

    counts = [event_count / len(corpus.sequences) for event_count in event_type_counts.values()]
    fig, ax = plt.subplots()
    ax.bar(event_types, counts)

    plt.bar(event_types, counts, align='center', alpha=0.5)
    plt.ylabel(f'Avg positions for different event types')
    plt.xticks(rotation=45)

    plt.savefig(f'{args.path["out_fold"]}/{file_name}.png')


def plot_sequence(corpus, seq_num, names=None, dates=None):
    if names is None or dates is None:
        names = corpus.sequences[seq_num].event_tokens[41:]
        dates = corpus.sequences[seq_num].event_times[41:]

    # Choose some nice levels
    levels = np.tile([-5, 5, -3, 3, -1, 1],
                     int(np.ceil(len(dates) / 6)))[:len(dates)]

    # Create figure and plot a stem plot with the date
    fig, ax = plt.subplots(figsize=(8.8, 4), constrained_layout=True)
    ax.set(title='Event Sequence')

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
    ax.get_xaxis().set_major_formatter(mdates.DateFormatter("%H:%M"))  # "%a %d - %H:%M"
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

    # Corpus configuration
    conf = {
        'task': 'binary',
        'binary_thresh': args.binary_thresh,
        'cats': args.categories,
        'years': args.years,
        'types': args.types,
        'seq_hours': args.seq_hours,
        'clip_los': args.clip_los
    }
    # Prepare corpus
    print('Loading Corpus')
    corpus = Corpus(
        data_path=args.path['data_fold'],
        file_names=args.file_names
    )
    vocab = corpus.prepare_corpus(conf)

    # Plot a sequence
    #dates, tokens = time_names()
    #plot_sequence(corpus, 0, tokens, dates)

    # Length of stay
    length_og_stay_plot(corpus, cutoff_days=14, file_name=f'los_bins_{conf["seq_hours"]}')
    length_og_stay_category_plot(corpus, label_names=['<2', '2-7', '7-14', '14-30', '30+'], file_name=f'los_cat_{conf["seq_hours"]}_hours')
    exit()

    # Sequence lengths
    hist_seq_len(corpus, bins=15, cutoff_len=256, file_name=f'seq_len_{conf["seq_hours"]}')

    # Token distribution
    token_distribution(corpus, bins=200, cutoff_min=0, cutoff_max=20000, file_name='token_dist_all')

    distinct_token_types(corpus, file_name=f'distinct_token_types_{conf["seq_hours"]}', event_types=conf["types"])

    # TODO: Investigate Apriori
    # Are there any apriori concepts that are not seen 200 times?

    apriori_distribution(corpus)

    # Token type distribution over time
    token_type_time_dist(corpus, file_name=f'tok_typ_dist_{conf["seq_hours"]}', event_types=conf["types"])

    # Token type distribution plot
    type_distribution(corpus, file_name=f'event_type_dist_{conf["seq_hours"]}', event_types=conf["types"])

    position_distribution(corpus, file_name=f'pos_dist_{conf["seq_hours"]}', event_types=conf["types"])
    exit()

    # TODO: Analyze the Aleatoric uncertainty of the problem by finding similar looking sequences
    # TODO: and look at the distribution in their LOS