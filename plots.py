import math
import os

from joblib import load
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchmetrics import ROC
import torch as th
import seaborn as sns


from Utils.utils import load_corpus, load_state_dict, set_seeds, load_baseline_date
from config import Config
from transformermodule.DataLoader.LOSLoader import HistoryLoader
from Utils.Evaluator import Evaluator
import matplotlib.pyplot as plt
from transformermodule.Model.MBERT import BertForMultiLabelPrediction
from transformermodule.Model.utils import BertConfig
from transformermodule.utils import get_model_config
from sklearn import metrics


def get_fpr_tpr(proba, labs, task='binary', tensor=False, num_classes=2):
    if task == 'category':
        task = 'multiclass'
        num_classes = 3

    if tensor:
        roc = ROC(task=task, num_classes=num_classes)
        fpr, tpr, _ = roc(proba, labs)
        fpr = fpr.cpu().numpy()
        tpr = tpr.cpu().numpy()
    else:
        fpr, tpr, _ = metrics.roc_curve(labs, proba)

    return fpr, tpr


def get_bert_model_and_data(conf):
    # Load corpus
    corpus = load_corpus(os.path.join(args.path['data_fold'], args.corpus_name))

    vocab = corpus.prepare_corpus(
        conf
    )
    corpus.create_pos_ids(event_dist=300)
    corpus.create_train_evel_test_idx(task=conf['task'], train_size=0.8)

    # Select subset corpus
    _, _, test_x = corpus.split_train_eval_test()

    # Setup dataloaders
    Dset = HistoryLoader(token2idx=vocab['token2index'], sequences=test_x, max_len=args.max_len_seq, conf=conf)
    testloader = DataLoader(dataset=Dset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # Create Bert Model
    model_config = get_model_config(vocab, args)
    feature_dict = {
        'word': True,
        'position': True,
        'age': True,
        'gender': True,
        'seg': False
    }
    bert_conf = BertConfig(model_config)

    # Setup Model
    model = BertForMultiLabelPrediction(
        args=args,
        bert_conf=bert_conf,
        feature_dict=feature_dict,
        cls_conf=conf,
        class_weights=None)

    model = model.to(args.device)

    # Initialize model parameters
    print(f'Loading state model with name: {conf["task"]}.pt')
    model = load_state_dict(f'{os.path.join(args.path["out_fold"], conf["task"])}.pt', model)

    return model, testloader


def evaluation(args, model, loader):
    model.eval()
    y_preds, y_label = None, None
    for step, (input_ids, posi_ids, age_ids, gender_ids, att_mask, labels, pat_ids) in enumerate(loader, 1):

        input_ids = input_ids.to(args.device)
        posi_ids = posi_ids.to(args.device)
        age_ids = age_ids.to(args.device)
        gender_ids = gender_ids.to(args.device)
        att_mask = att_mask.to(args.device)
        labels = labels.to(args.device)

        with th.no_grad():
            loss, logits = model(input_ids, posi_ids, age_ids, gender_ids, targets=labels, attention_mask=att_mask)

        y_preds = logits if y_preds is None else th.cat((y_preds, logits))
        y_label = labels if y_label is None else th.cat((y_label, labels))

    return y_preds, y_label


def evaluate_chunks(args, model, loader):
    model.eval()
    age_chunk_dict = {}
    gender_chunk_dict = {}
    for step, (input_ids, posi_ids, age_ids, gender_ids, att_mask, labels, pat_ids) in enumerate(loader, 1):

        input_ids = input_ids.to(args.device)
        posi_ids = posi_ids.to(args.device)
        age_ids = age_ids.to(args.device)
        gender_ids = gender_ids.to(args.device)
        att_mask = att_mask.to(args.device)
        labels = labels.to(args.device)

        with th.no_grad():
            loss, logits = model(input_ids, posi_ids, age_ids, gender_ids, targets=labels, attention_mask=att_mask)

        for log, age, lab in zip(logits, age_ids[:, 0], labels):
            age_group = int(math.ceil(float(age.to('cpu').numpy() / 10.0)) * 10)
            if age_group in age_chunk_dict:
                age_chunk_dict[age_group][0] = th.cat((age_chunk_dict[age_group][0], log))
                age_chunk_dict[age_group][1] = th.cat((age_chunk_dict[age_group][1], lab))
            else:
                age_chunk_dict[age_group] = [log, lab]

        for log, gender, lab in zip(logits, gender_ids[:, 0], labels):
            gender_group = int(gender.to('cpu').numpy())
            if gender_group in gender_chunk_dict:
                gender_chunk_dict[gender_group][0] = th.cat((gender_chunk_dict[gender_group][0], log))
                gender_chunk_dict[gender_group][1] = th.cat((gender_chunk_dict[gender_group][1], lab))
            else:
                gender_chunk_dict[gender_group] = [log, lab]

    return age_chunk_dict, gender_chunk_dict


if __name__ == '__main__':
    config_file = ['config.ini']
    args = Config(
        file_path=config_file
    )
    set_seeds(42)

    conf = {
        'task': 'binary',
        'metrics': ['auc'],
        'binary_thresh': 2,
        'cats': [2, 7],
        'years': [2018, 2019, 2020, 2021],
        'types': ['apriori', 'adm', 'proc', 'vital', 'lab'],
        'max_hours': 24
    }
    nn_model, rfc_model, svc_model = None, None, None
    # Load all the sklearn models
    if conf['task'] == 'binary':
        nn_model = load(os.path.join(args.path['out_fold'], 'nn_binary.joblib'))
        rfc_model = load(os.path.join(args.path['out_fold'], 'rfc_binary.joblib'))
        svc_model = load(os.path.join(args.path['out_fold'], 'svc_binary.joblib'))

    elif conf['task'] == 'category':
        nn_model = load(os.path.join(args.path['out_fold'], 'nn_category.joblib'))
        rfc_model = load(os.path.join(args.path['out_fold'], 'rfc_category.joblib'))
        svc_model = load(os.path.join(args.path['out_fold'], 'svc_category.joblib'))

    # Load the trained bert models
    th_model, th_loader = get_bert_model_and_data(conf)

    # Load the test data for sklearn models
    _, _, test_x, test_y = load_baseline_date(args.path['out_fold'], conf['task'])

    # Create evaluator
    evaluator = Evaluator(conf=conf, device=args.device)

    # Create prediction chunks
    age_rfc_fprs = {}
    gender_rfc_fprs = {}
    age_aucs = {}
    gender_aucs = {}
    age_chunks, gender_chunks = evaluate_chunks(args, th_model, th_loader)
    age_chunks = dict(sorted(age_chunks.items()))
    for name, chunk in age_chunks.items():
        if len(chunk[0]) > 100:
            age_rfc_fprs[name] = get_fpr_tpr(chunk[0], chunk[1], task=conf['task'], tensor=True)
            age_aucs[name] = evaluator.calculate_metrics(chunk[0], chunk[1])['auc']

    for name, chunk in gender_chunks.items():
        gender_rfc_fprs[name] = get_fpr_tpr(chunk[0], chunk[1], task=conf['task'], tensor=True)
        gender_aucs[name] = evaluator.calculate_metrics(chunk[0], chunk[1])['auc']

    # Create probabilities
    th_proba, th_test_y = evaluation(args, th_model, th_loader)
    rfc_proba = rfc_model.predict_proba(test_x)[:, 1]
    nn_proba = nn_model.predict_proba(test_x)[:, 1]
    svc_proba = svc_model.predict_proba(test_x)[:, 1]

    # Get model metrics
    th_auc = evaluator.calculate_metrics(th_proba, th_test_y)['auc']
    rfc_auc = roc_auc_score(test_y, rfc_proba, multi_class='ovo')
    nn_auc = roc_auc_score(test_y, nn_proba, multi_class='ovo')
    svc_auc = roc_auc_score(test_y, svc_proba, multi_class='ovo')

    # Extract fpr and tpr for each example
    th_fpr, th_tpr = get_fpr_tpr(th_proba, th_test_y, tensor=True)
    rfc_fpr, rfc_tpr = get_fpr_tpr(rfc_proba, test_y)
    nn_fpr, nn_tpr = get_fpr_tpr(nn_proba, test_y)
    svc_fpr, svc_tpr = get_fpr_tpr(svc_proba, test_y)

    # create ROC curve
    sns.set_theme()
    fig, axs = plt.subplots()
    plt.plot(rfc_fpr, rfc_tpr, label=f'RFC AUROC = {round(rfc_auc, 2)}')
    plt.plot(nn_fpr, nn_tpr, label=f'ANN AUROC = {round(nn_auc, 2)}')
    plt.plot(svc_fpr, svc_tpr, label=f'SVC AUROC = {round(svc_auc, 2)}')
    plt.plot(th_fpr, th_tpr, label=f'M-BERT AUROC = {round(th_auc, 2)}')
    plt.plot([0, 1], [0, 1], color='orange', linestyle='--')
    axs.set_ylabel('True Positive Rate', fontsize=18)
    axs.tick_params(axis='both', which='major', labelsize=14)
    axs.set_xlabel('False Positive Rate', fontsize=18)
    fig.subplots_adjust(left=0.15, bottom=0.15)

    plt.legend(loc='lower right', fontsize=14)

    plt.savefig(f'{args.path["out_fold"]}/roc_{conf["task"]}.eps', format='eps')
    plt.clf()



    # Create ROC curve for ages
    fig, axs = plt.subplots()
    for rfc_fpr, auc in zip(age_rfc_fprs.items(), age_aucs.items()):
        min_years = int(auc[0])
        max_years = min_years + 10

        plt.plot(rfc_fpr[1][0], rfc_fpr[1][1], label=f'AUROC {min_years}-{max_years} = {round(auc[1], 2)}')

    plt.plot([0, 1], [0, 1], color='orange', linestyle='--')
    axs.set_ylabel('True Positive Rate', fontsize=18)
    axs.tick_params(axis='both', which='major', labelsize=14)
    axs.set_xlabel('False Positive Rate', fontsize=18)
    fig.subplots_adjust(left=0.15, bottom=0.15)

    plt.legend(loc='lower right', fontsize=14)

    plt.savefig(f'{args.path["out_fold"]}/roc_{conf["task"]}_ages.eps', format='eps')
    plt.clf()



    # Create ROC curve for genders
    for rfc_fpr, gender in zip(gender_rfc_fprs.items(), gender_aucs.items()):
        sex = 'Male' if int(gender[0]) == 0 else 'Female'

        plt.plot(rfc_fpr[1][0], rfc_fpr[1][1], label=f'AUROC {sex} = {round(gender[1], 2)}')

    plt.plot([0, 1], [0, 1], color='orange', linestyle='--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc='lower right')

    plt.savefig(f'{args.path["out_fold"]}/roc_{conf["task"]}_gender.png', format='png')
    plt.clf()

