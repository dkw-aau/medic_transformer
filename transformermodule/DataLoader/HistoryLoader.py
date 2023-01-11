import numpy as np
from bisect import bisect
from torch.utils.data.dataset import Dataset
from transformermodule.utils import seq_padding, code2index, limit_seq_length
import torch as th


class HistoryLoader(Dataset):
    def __init__(self, token2idx, sequences, max_len, conf=None, tasks=None):
        # TODO: Move all data to device in init (Potential train speedup?)
        self.vocab = token2idx
        self.max_len = max_len
        self.sequences = sequences
        self.conf = conf
        self.tasks = tasks

        self.codes = []
        self.masks = []
        self.labels = []
        self.posi = []

        # format sequences
        for seq in self.sequences:
            # TODO: Move task to some other place...
            labs = {}
            if 'los_real' in self.tasks:
                labs['los_real'] = th.tensor(seq.length_of_stay, dtype=th.float32)
            if 'los_binary' in self.tasks:
                labs['los_binary'] = th.tensor([1 if seq.length_of_stay > self.conf['los_binary_threshold'] else 0],
                                               dtype=th.float32)
            if 'los_category' in self.tasks:
                labs['los_category'] = th.tensor([bisect(self.conf['classes'], seq.length_of_stay)], dtype=th.long)
            if 'mortality_30' in self.tasks:
                labs['los_binary'] = th.tensor(1 if seq.mortality_30 else 0, dtype=th.float32)
            if 'hosp_binary' in self.tasks:
                labs['los_binary'] = th.tensor(1 if seq.req_hosp else 0, dtype=th.float32)

            code = limit_seq_length(seq.event_tokens, self.max_len)

            mask = np.ones(self.max_len)
            mask[len(code):] = 0

            # pad code sequence
            tokens, code = code2index(code, self.vocab)

            # Positions
            pos_start = seq.event_pos_ids[:len(code)]
            pos = pos_start + [pos_start[-1] + 1] * max(0, self.max_len - len(pos_start))

            # pad code and label
            code = seq_padding(code, self.max_len, symbol=self.vocab['PAD'])

            self.codes.append(code)
            self.masks.append(mask)
            self.labels.append(labs)
            self.posi.append(pos)

    def __getitem__(self, idx):
        """
        return: code, mask, label
        """
        return th.LongTensor(self.codes[idx]), th.LongTensor(self.posi[idx]), th.LongTensor(self.masks[idx]), self.labels[idx], th.LongTensor([idx])

    def __len__(self):
        return len(self.sequences)
