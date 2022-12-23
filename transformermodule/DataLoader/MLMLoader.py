from torch.utils.data.dataset import Dataset
import numpy as np
from transformermodule.utils import seq_padding, position_idx, random_mask, limit_seq_length
import torch as th


# TODO: Make typed
class MLMLoader(Dataset):
    def __init__(self, sequences, token2idx, max_len):
        self.vocab = token2idx
        self.max_len = max_len
        self.sequences = sequences

    def __getitem__(self, index):
        """
        return: code, position, mask, label
        """
        # extract data
        seq = self.sequences[index]
        code = limit_seq_length(seq.event_tokens, self.max_len, seq.apriori_len)

        # mask 0:len(code) to 1, padding to be 0
        mask = np.ones(self.max_len)
        mask[len(code):] = 0

        # pad sequence and code sequence
        tokens, code, label = random_mask(code, self.vocab)

        # get position code and segment code
        tokens = seq_padding(tokens, self.max_len)
        position = position_idx(tokens)

        # pad code and label
        code = seq_padding(code, self.max_len, symbol=self.vocab['PAD'])
        label = seq_padding(label, self.max_len, symbol=-1)

        return th.LongTensor(code), th.LongTensor(position), th.LongTensor(mask), th.LongTensor(label)

    def __len__(self):
        return len(self.sequences)
