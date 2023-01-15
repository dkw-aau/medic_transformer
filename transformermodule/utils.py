import random


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


def get_model_config(vocab, args):
    return {
        'vocab_size': len(vocab['token2index'].keys()),  # num embeddings
        'hidden_size': args.hidden_size,  # word embedding and index embedding hidden size
        'max_position_embedding': args.max_len_seq,  # maximum number of tokens
        'hidden_dropout_prob': args.layer_dropout,  # dropout rate
        'num_hidden_layers': args.num_hidden_layers,  # number of multi-head attention layers required
        'num_attention_heads': args.num_attention_heads,  # number of attention heads
        'attention_probs_dropout_prob': args.att_dropout,  # multi-head attention dropout rate
        'intermediate_size': args.intermediate_size,  # the size of the "intermediate" layer in the transformer encoder
        'hidden_act': args.hidden_act,
        # The non-linear activation function in the encoder and the pooler "gelu", 'relu', 'swish' are supported
        'initializer_range': args.initializer_range,  # parameter weight initializer range
    }