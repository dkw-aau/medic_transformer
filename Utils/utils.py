import os
import pickle
import random
import time

import numpy as np
import pandas as pd
import torch as th


def save_model(save_path, model):
    model.eval()
    th.save(model.state_dict(), save_path)


def save_model_state(model, file_path):
    print("*** Saving model ***")
    th.save(model.state_dict(), file_path)


def pickle_save(data, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)


def pickle_load(save_path):
    with open(os.path.join(save_path), 'rb') as f:
        return pickle.load(f)


def save_corpus(corpus, file_name):
    corpus.corpus_df.to_parquet(f'{file_name}.parquet')
    corpus.corpus_df = None
    pickle_save(corpus, f'{file_name}.pkl')


def load_corpus(file_name):
    print('Loading Corpus')
    start = time.time()
    corpus = pickle_load(f'{file_name}.pkl')
    print(f'load_pickle took: {time.time() - start}')
    start = time.time()
    corpus_df = pd.read_parquet(f'{file_name}.parquet')
    print(f'load_corpus_df took: {time.time() - start}')
    start = time.time()
    corpus.corpus_df = corpus_df
    corpus.unpack_corpus_df()
    print(f'unpacking corpus_df took: {time.time() - start}')

    return corpus


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def save_baseline_data(train_x, train_y, test_x, test_y, task, path):
    print('Saving train and test data to file')
    train = np.append(train_x, np.expand_dims(train_y, axis=1), axis=1)
    test = np.append(test_x, np.expand_dims(test_y, axis=1), axis=1)
    with open(os.path.join(path, f'base_train_{task}.npy'), 'wb') as f:
        np.save(f, train)
    with open(os.path.join(path, f'base_test_{task}.npy'), 'wb') as f:
        np.save(f, test)


def load_baseline_date(path, task):
    with open(os.path.join(path, f'base_train_{task}.npy'), 'rb') as f:
        train = np.load(f)
    with open(os.path.join(path, f'base_test_{task}.npy'), 'rb') as f:
        test = np.load(f)

    train_x = train[:, :-1]
    train_y = train[:, -1]
    test_x = test[:, :-1]
    test_y = test[:, -1]

    return train_x, train_y, test_x, test_y


def set_seeds(seed):
    random.seed(seed)
    th.manual_seed(1234)


def load_state_dict(path, model):
    # load pretrained model and update weights
    pretrained_dict = th.load(path, map_location='cpu')
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)

    return model

