import numpy as np
from torch.utils.data.dataset import Dataset
from DataLoader.utils import seq_padding, code2index, position_idx, index_seg, limit_seq_length
import torch as th


class NextVisit(Dataset):
    def __init__(self, token2idx, sequences, max_len):
        # dataframe preproecssing
        # filter out the patient with number of visits less than min_visit
        self.vocab = token2idx
        self.max_len = max_len
        self.sequences = sequences

    def __getitem__(self, index):
        """
        return: age, code, position, segmentation, mask, label
        """
        # cut data
        seq = self.sequences[index]
        code = limit_seq_length(seq.event_tokens, self.max_len, seq.apriori_len)

        label = {
            'los': th.tensor(seq.length_of_stay, dtype=th.float32),
            'm30': th.tensor(1 if seq.mortality_30 else 0, dtype=th.float32),
            'hosp': th.tensor(1 if seq.req_hosp else 0, dtype=th.float)
        }

        pat_id = seq.pat_id

        # mask 0:len(code) to 1, padding to be 0
        mask = np.ones(self.max_len)
        mask[len(code):] = 0

        # pad code sequence
        tokens, code = code2index(code, self.vocab)

        # get position code and segment code
        tokens = seq_padding(tokens, self.max_len)
        position = position_idx(tokens)

        # pad code and label
        code = seq_padding(code, self.max_len, symbol=self.vocab['PAD'])

        return th.LongTensor(code), th.LongTensor(position), \
               th.LongTensor(mask), label, th.LongTensor([int(pat_id)])

    def __len__(self):
        return len(self.sequences)
