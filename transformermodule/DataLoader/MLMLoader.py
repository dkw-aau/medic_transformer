import numpy as np
import torch as th
from torch.utils.data.dataset import Dataset

from Utils.utils import seq_padding, random_mask, limit_seq_length, pad_position


class MLMLoader(Dataset):
    def __init__(self, token2idx, sequences, max_len):
        self.vocab = token2idx
        self.max_len = max_len
        self.sequences = sequences

    def __getitem__(self, idx):
        """
        return: code, mask, label
        """
        seq = self.sequences[idx]
        code = limit_seq_length(seq.event_tokens, self.max_len)

        mask = np.ones(self.max_len)
        mask[len(code):] = 0

        # pad sequence and code sequence
        tokens, code, label = random_mask(code, self.vocab)

        # pad position sequence
        pos = seq.event_pos_ids[:len(code)]
        pos = pad_position(pos, self.max_len)

        # pad code and label
        code = seq_padding(code, self.max_len, symbol=self.vocab['PAD'])
        label = seq_padding(label, self.max_len, symbol=-1)

        sex = [seq.sex] * self.max_len
        age = [seq.age] * self.max_len

        return th.LongTensor(code), th.LongTensor(pos), th.LongTensor(age), th.LongTensor(sex), \
               th.LongTensor(mask), th.LongTensor(label), th.LongTensor([idx])

    def __len__(self):
        return len(self.sequences)
