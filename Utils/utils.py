import os
import pickle
import random
import time
import pytorch_pretrained_bert as Bert

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


class BertConfig(Bert.modeling.BertConfig):
    def __init__(self, config):
        super(BertConfig, self).__init__(
            vocab_size_or_config_json_file=config.get('vocab_size'),
            hidden_size=config['hidden_size'],
            num_hidden_layers=config.get('num_hidden_layers'),
            num_attention_heads=config.get('num_attention_heads'),
            intermediate_size=config.get('intermediate_size'),
            hidden_act=config.get('hidden_act'),
            hidden_dropout_prob=config.get('hidden_dropout_prob'),
            attention_probs_dropout_prob=config.get('attention_probs_dropout_prob'),
            max_position_embeddings=config.get('max_position_embedding'),
            initializer_range=config.get('initializer_range'),
        )


def get_model_config(vocab, args):
    return {
        'vocab_size': len(vocab['token2index'].keys()),  # num embeddings
        'hidden_size': args['hidden_size'],  # word embedding and index embedding hidden size
        'max_position_embedding': args['max_len_seq'],  # maximum number of tokens
        'hidden_dropout_prob': args['layer_dropout'],  # dropout rate
        'num_hidden_layers': args['num_hidden_layers'],  # number of multi-head attention layers required
        'num_attention_heads': args['num_attention_heads'],  # number of attention heads
        'attention_probs_dropout_prob': args['att_dropout'],  # multi-head attention dropout rate
        'intermediate_size': args['intermediate_size'],  # the size of the "intermediate" layer in the transformer encoder
        'hidden_act': args['hidden_act'],
        # The non-linear activation function in the encoder and the pooler "gelu", 'relu', 'swish' are supported
        'initializer_range': args['initializer_range'],  # parameter weight initializer range
    }


def load_model(path, model):
    # load pretrained model and update weights
    pretrained_dict = th.load(path)
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    return model


def code2index(tokens, token2idx, mask_token=None):
    output_tokens = []
    for i, token in enumerate(tokens):
        if token == mask_token:
            output_tokens.append(token2idx['UNK'])
        else:
            output_tokens.append(token2idx.get(token, token2idx['UNK']))
    return tokens, output_tokens


def pad_position(pos, max_len):
    last_pos = 0 if len(pos) == 0 else pos[-1] + 1
    pos = pos + [last_pos] * max(0, max_len - len(pos))
    return pos


def random_mask(tokens, token2idx):
    output_label = []
    output_token = []
    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                output_token.append(token2idx['MASK'])

            # 10% randomly change token to random token
            elif prob < 0.9:
                output_token.append(random.choice(list(token2idx.values())))

            # -> rest 10% randomly keep current token
            # append current token to output (we will predict these later)
            output_label.append(token2idx.get(token, token2idx['UNK']))
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)
            output_token.append(token2idx.get(token, token2idx['UNK']))

    return tokens, output_token, output_label


def limit_seq_length(seq, max_len):
    if len(seq) > max_len:
        return seq[:max_len]
    else:
        return seq


def index_seg(tokens, apriori_len):
    seg = [0] * apriori_len + [1] * (len(tokens) - apriori_len)

    return seg


def position_idx(tokens):
    pos = [x for x in range(0, len(tokens))]
    return pos


def seq_padding(tokens, max_len, token2idx=None, symbol=None, unkown=True):
    if symbol is None:
        symbol = 'PAD'

    seq = []
    token_len = len(tokens)
    for i in range(max_len):
        if token2idx is None:
            if i < token_len:
                seq.append(tokens[i])
            else:
                seq.append(symbol)
        else:
            if i < token_len:
                # 1 indicate UNK
                if unkown:
                    seq.append(token2idx.get(tokens[i], token2idx['UNK']))
                else:
                    seq.append(token2idx.get(tokens[i]))
            else:
                seq.append(token2idx.get(symbol))
    return seq

