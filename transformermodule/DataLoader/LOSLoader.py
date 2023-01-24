import numpy as np
from bisect import bisect
from torch.utils.data.dataset import Dataset
from transformermodule.utils import seq_padding, code2index, limit_seq_length, pad_position
import torch as th


class HistoryLoader(Dataset):
    def __init__(self, token2idx, sequences, max_len, task=None, scaler=None):
        # TODO: Move all data to device in init (Potential train speedup?)
        self.vocab = token2idx
        self.max_len = max_len
        self.sequences = sequences
        self.task = task
        self.scaler = scaler

        self.codes = []
        self.masks = []
        self.labels = []
        self.posi = []
        self.ages = []
        self.genders = []

        # format sequences
        for seq in self.sequences:

            code = limit_seq_length(seq.event_tokens, self.max_len)
            if self.task == 'real':
                label = float(self.scaler(np.array(seq.label).reshape(-1, 1))[0])
            else:
                label = seq.label
            mask = np.ones(self.max_len)
            mask[len(code):] = 0

            # pad code sequence
            tokens, code = code2index(code, self.vocab)

            # pad position sequence
            pos = seq.event_pos_ids[:len(code)]
            pos = pad_position(pos, self.max_len)

            # pad code and label
            code = seq_padding(code, self.max_len, symbol=self.vocab['PAD'])

            gender = [seq.gender] * self.max_len
            age = [seq.age] * self.max_len

            self.codes.append(code)
            self.masks.append(mask)
            self.labels.append(label)
            self.posi.append(pos)

            self.genders.append(gender)
            self.ages.append(age)

    def __getitem__(self, idx):
        """
        return: code, mask, label
        """
        return th.LongTensor(self.codes[idx]), \
               th.LongTensor(self.posi[idx]), \
               th.LongTensor(self.ages[idx]), \
               th.LongTensor(self.genders[idx]), \
               th.LongTensor(self.masks[idx]), \
               th.LongTensor([self.labels[idx]]) if self.task == 'category' else th.FloatTensor([self.labels[idx]]), \
               th.LongTensor([idx])

    def __len__(self):
        return len(self.sequences)
