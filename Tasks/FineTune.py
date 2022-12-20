from multiprocessing import freeze_support

from Evaluation.Evaluator import Evaluator
from config import get_file_config, get_global_params, get_train_params, get_model_config, \
    get_optim_config
from Model.LengthOfStay import BertForMultiLabelPrediction
from torchmetrics.classification import BinaryPrecision
from torchmetrics import MeanAbsoluteError
from DataLoader.LOSLoader import NextVisit
from torch.utils.data import DataLoader
from Common.common import create_folder
from Model.utils import BertConfig
from Common.utils import load_corpus, load_state_dict, save_model_state
from Model import optimiser
from tqdm import tqdm
import torch as th
import warnings
import torch
import time
import os


warnings.filterwarnings(action='ignore')

# Setup parameters
file_config = get_file_config()
global_params = get_global_params()
train_params = get_train_params()
optim_config = get_optim_config()

corpus = load_corpus(os.path.join(file_config['path'], file_config['corpus']))

vocab = corpus.vocabulary
train, evalu, test = corpus.get_data_split()

model_config = get_model_config(vocab, train_params)

feature_dict = {
    'word': True,
    'position': True
}

# Train Data
Dset = NextVisit(token2idx=vocab['token2index'], sequences=train, max_len=train_params['max_len_seq'])
trainloader = DataLoader(dataset=Dset, batch_size=train_params['batch_size'], shuffle=True, num_workers=0)

Dset = NextVisit(token2idx=vocab['token2index'], sequences=test, max_len=train_params['max_len_seq'])
testloader = DataLoader(dataset=Dset, batch_size=train_params['batch_size'], shuffle=False, num_workers=0)

conf = BertConfig(model_config)
# los_real, req_hosp, mortality_30
tasks = ['mortality_30']
# mae, prec
metrics = ['prec']
evaluator = Evaluator(tasks=tasks, metrics=metrics)
model = BertForMultiLabelPrediction(
    config=conf,
    feature_dict=feature_dict,
    cls_heads=tasks,
    cls_config=None)

model = load_state_dict(os.path.join(file_config['output_path'], file_config['pretrain_name']), model)
model = model.to(train_params['device'])
optim = optimiser.adam(params=list(model.named_parameters()), config=optim_config)


def train(e):
    model.train()
    tr_loss = 0
    tr_prec = 0
    loader_iter = tqdm(trainloader)
    for step, batch in enumerate(loader_iter, 1):
        step_time = time.time()
        input_ids, posi_ids, att_mask, targets, pat_ids = batch

        input_ids = input_ids.to(train_params['device'])
        posi_ids = posi_ids.to(train_params['device'])
        att_mask = att_mask.to(train_params['device'])
        targets = {key: lab.to(train_params['device']) for key, lab in targets.items()}

        loss, logits = model(input_ids, posi_ids, targets=targets, attention_mask=att_mask)

        if global_params['gradient_accumulation_steps'] > 1:
            loss = loss / global_params['gradient_accumulation_steps']
        loss.backward()

        metrics = evaluator.calculate_metrics(logits, targets)

        tr_loss += loss.item()
        print_dict = {'epoch': e, 'loss': tr_loss / step, 'time': time.time() - step_time}
        print_dict.update(metrics)
        loader_iter.set_postfix(print_dict)

        if (step + 1) % global_params['gradient_accumulation_steps'] == 0:
            optim.step()
            optim.zero_grad()


def evaluation():
    model.eval()
    y_preds = None
    y_label = None
    tr_loss = 0
    loader_iter = tqdm(testloader)
    for step, batch in enumerate(loader_iter, 1):
        model.eval()
        input_ids, posi_ids, att_mask, targets, pat_ids = batch

        input_ids = input_ids.to(train_params['device'])
        posi_ids = posi_ids.to(train_params['device'])
        att_mask = att_mask.to(train_params['device'])
        targets = {key: lab.to(train_params['device']) for key, lab in targets.items()}

        with torch.no_grad():
            loss, logits = model(input_ids, posi_ids, targets=targets, attention_mask=att_mask)

        tr_loss += loss.item()

        y_preds = logits if y_preds is None else th.cat((y_preds, logits))
        y_label = targets if y_label is None else th.cat((y_label, targets['hosp']))

    metrics = evaluator.calculate_metrics(y_preds, y_label)

    return tr_loss / step, metrics


if __name__ == '__main__':
    freeze_support()
    best_loss = 100000
    for e in range(50):
        epoch_time = time.time()
        train(e)
        loss, metrics = evaluation()
        print(f'Epoch: {e}, Eval Loss: {loss:.3}, Epoch Time: {time.time() - epoch_time}, {metrics} ')
        time.sleep(1)

        if loss < best_loss:
            save_model_state(model, file_config['output_path'], file_config['finetune_name'])
            best_loss = loss

        print(f'Overall Best Prec: {best_loss:.3}')
