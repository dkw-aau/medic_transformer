import numpy as np
from bisect import bisect
from torch.utils.data.dataset import Dataset
from transformermodule.utils import seq_padding, code2index, limit_seq_length, index_seg
import torch as th


class LOSLoader(Dataset):
    def __init__(self, token2idx, sequences, max_len, conf=None, tasks=None):
        # dataframe preproecssing
        # filter out the patient with number of visits less than min_visit
        self.vocab = token2idx
        self.max_len = max_len
        self.sequences = sequences
        self.conf = conf
        self.tasks = tasks

    def __getitem__(self, index):
        """
        return: age, code, position, segmentation, mask, label
        """
        # cut data
        seq = self.sequences[index]
        code = limit_seq_length(seq.event_tokens, self.max_len)

        labels = {}

        if 'los_real' in self.tasks:
            labels['los_real'] = th.tensor(seq.length_of_stay, dtype=th.float32)
        if 'los_binary' in self.tasks:
            labels['los_binary'] = th.tensor([1 if seq.length_of_stay > self.conf['los_binary_threshold'] else 0], dtype=th.float32)
        if 'los_category' in self.tasks:
            labels['los_category'] = th.tensor([bisect(self.conf['classes'], seq.length_of_stay)], dtype=th.long)
        if 'mortality_30' in self.tasks:
            labels['los_binary'] = th.tensor(1 if seq.mortality_30 else 0, dtype=th.float32)
        if 'hosp_binary' in self.tasks:
            labels['los_binary'] = th.tensor(1 if seq.req_hosp else 0, dtype=th.float32)

        pat_id = seq.pat_id

        # mask 0:len(code) to 1, padding to be 0
        mask = np.ones(self.max_len)
        mask[len(code):] = 0

        # pad code sequence
        tokens, code = code2index(code, self.vocab)

        # get position code and segment code
        tokens = seq_padding(tokens, self.max_len)
        #position = position_idx(tokens)
        pos_start = seq.event_pos_ids[:len(code)]
        position = pos_start + [pos_start[-1] + 1] * max(0, self.max_len - len(pos_start))
        segment = index_seg(tokens, seq.apriori_len)

        # pad code and label
        code = seq_padding(code, self.max_len, symbol=self.vocab['PAD'])

        return th.LongTensor(code), th.LongTensor(position), th.LongTensor(segment), \
               th.LongTensor(mask), labels, th.LongTensor([int(pat_id)])

    def __len__(self):
        return len(self.sequences)
