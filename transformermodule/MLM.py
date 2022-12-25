from multiprocessing import freeze_support
from torch.utils.data import DataLoader
from .Model.MLMModel import BertForMaskedLM
from .utils import get_model_config
from .DataLoader.MLMLoader import MLMLoader
from .Model.utils import BertConfig
from .Model.optimiser import adam
import sklearn.metrics as skm
from Utils.utils import load_corpus, save_model_state, create_folder
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import warnings
import torch
import time
import os


class MLMTrainer:
    def __init__(
            self,
             args):

        self.args = args
        freeze_support()
        warnings.filterwarnings(action='ignore')
        create_folder(args.path['out_fold'])

        # Load corpus
        corpus = load_corpus(os.path.join(args.path['data_fold'], args.corpus_name))
        vocab = corpus.vocabulary
        train, _, _ = corpus.get_data_split()

        # Setup dataloader
        Dset = MLMLoader(train, vocab['token2index'], max_len=args.max_len_seq)
        self.trainload = DataLoader(dataset=Dset, batch_size=args.batch_size, shuffle=True, num_workers=0)

        # Create Bert Model
        model_config = get_model_config(vocab, args)
        conf = BertConfig(model_config)
        model = BertForMaskedLM(conf)
        self.model = model.to(args.device)
        self.optim = adam(params=list(model.named_parameters()), args=args)

    def train(self, epochs):
        for e in range(0, epochs):
            self.epoch(e)

    def epoch(self, e):
        tr_loss = 0
        epoch_time = time.time()

        loader_iter = tqdm(self.trainload)
        for step, batch in enumerate(loader_iter, 1):
            step_time = time.time()
            batch = tuple(t.to(self.args.device) for t in batch)
            input_ids, posi_ids, attMask, masked_label = batch
            loss, pred, label = self.model(input_ids, posi_ids, attention_mask=attMask, masked_lm_labels=masked_label)

            loss.backward()

            tmp_loss = loss.item()
            tr_loss += tmp_loss

            # prec = cal_acc(label, pred)
            loader_iter.set_postfix({'epoch': e, 'loss': tmp_loss, 'time': time.time() - step_time})

            self.optim.step()
            self.optim.zero_grad()

        save_model_state(self.model, self.args.path['out_fold'], self.args.pretrain_name)

        return tr_loss / step,  time.time() - epoch_time
