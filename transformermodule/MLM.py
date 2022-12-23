from multiprocessing import freeze_support
from torch.utils.data import DataLoader
from .Model.MLMModel import BertForMaskedLM
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


def cal_acc(label, pred):
    logs = nn.LogSoftmax()
    label = label.cpu().numpy()
    ind = np.where(label != -1)[0]
    truepred = pred.detach().cpu().numpy()
    truepred = truepred[ind]
    truelabel = label[ind]
    truepred = logs(torch.tensor(truepred))
    outs = [np.argmax(pred_x) for pred_x in truepred.numpy()]
    precision = skm.precision_score(truelabel, outs, average='micro')
    return precision


def train(e, loader):
    tr_loss = 0
    epoch_time = time.time()

    loader_iter = tqdm(loader)
    for step, batch in enumerate(loader_iter, 1):
        step_time = time.time()
        batch = tuple(t.to(train_params['device']) for t in batch)
        input_ids, posi_ids, attMask, masked_label = batch
        loss, pred, label = model(input_ids, posi_ids, attention_mask=attMask, masked_lm_labels=masked_label)

        if global_params['gradient_accumulation_steps'] > 1:
            loss = loss / global_params['gradient_accumulation_steps']
        loss.backward()

        tmp_loss = loss.item()
        tr_loss += tmp_loss

        prec = cal_acc(label, pred)
        loader_iter.set_postfix({'epoch': e, 'loss': tmp_loss, 'prec': prec, 'time': time.time() - step_time})

        if (step + 1) % global_params['gradient_accumulation_steps'] == 0:
            optim.step()
            optim.zero_grad()

    save_model_state(model, file_config['output_path'], file_config['pretrain_name'])

    return tr_loss / step,  time.time() - epoch_time


def write_log(text):
    f = open(os.path.join(file_config['output_path'], 'log_mlm.txt'), "a")
    f.write(text)
    f.close()


def train_mlm(args):
    freeze_support()

    warnings.filterwarnings(action='ignore')

    create_folder(args.path['out_fold'])

    # Load corpus
    corpus = load_corpus(os.path.join(args.path['data_fold'], args.corpus_name))

    vocab = corpus.vocabulary
    train, _, _ = corpus.get_data_split()

    Dset = MLMLoader(train, vocab['token2index'], max_len=train_params['max_len_seq'])
    trainload = DataLoader(dataset=Dset, batch_size=train_params['batch_size'], shuffle=True, num_workers=0)

    model_config = get_model_config(vocab, train_params)

    # Create Bert Model
    conf = BertConfig(model_config)
    model = BertForMaskedLM(conf)

    model = model.to(train_params['device'])
    optim = adam(params=list(model.named_parameters()), config=optim_config)

    write_log('{}\t{}\t{}\n'.format('epoch', 'loss', 'time'))
    for e in range(50):
        loss, time_cost = train(e, trainload)
        print(f'Epoch: {e}, Loss: {loss}, Time: {time_cost}')
        write_log('{}\t{}\t{}\n'.format(e, loss, time_cost))
