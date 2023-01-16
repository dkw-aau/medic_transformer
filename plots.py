import os

from joblib import dump, load
from torch.utils.data import DataLoader
from torchmetrics import ROC
import torch as th

from Utils.utils import load_baseline_date, load_corpus, load_state_dict
from config import Config
from transformermodule.DataLoader.HistoryLoader import HistoryLoader
from transformermodule.Evaluation.Evaluator import Evaluator
import matplotlib.pyplot as plt
from sklearn import metrics
from transformermodule.Model.LengthOfStay import BertForMultiLabelPrediction
from transformermodule.Model.utils import BertConfig
from transformermodule.utils import get_model_config


def get_bert_model_and_data(conf):
    # Load corpus
    corpus = load_corpus(os.path.join(args.path['data_fold'], args.corpus_name))

    vocab = corpus.prepare_corpus(
        conf
    )
    corpus.create_pos_ids(event_dist=300)
    corpus.create_train_evel_test_idx(train_size=0.8)

    # Select subset corpus
    _, _, test_x = corpus.split_train_eval_test()
    _, _, test_y = corpus.split_train_eval_test_labels()

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
    print(f'Loading state model with name: {args.load_name}')
    model = load_state_dict(os.path.join(args.path['out_fold'], f'{conf["task"]}_full.pt'), model)

    return model, testloader


def evaluation(args, model, loader):
    model.eval()
    y_preds, y_label = None, None
    for step, (input_ids, posi_ids, age_ids, gender_ids, att_mask, labels, pat_ids) in enumerate(loader, 1):
        model.eval()

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


if __name__ == '__main__':
    config_file = ['config.ini']
    args = Config(
        file_path=config_file
    )

    conf = {
        'task': 'category',
        'metric': 'f1',
        'binary_thresh': 2,
        'cats': [2, 7],
        'years': [2018, 2019, 2020, 2021],
        'types': ['apriori', 'adm', 'proc', 'vital', 'lab'],
        'max_hours': 24
    }

    # Load all the sklearn models
    #nn_binary = load(os.path.join(args.path['out_fold'], 'nn_binary.joblib'))
    #rfc_binary = load(os.path.join(args.path['out_fold'], 'rfc_binary.joblib'))
    #svc_binary = load(os.path.join(args.path['out_fold'], 'svc_binary.joblib'))
    #nn_category = load(os.path.join(args.path['out_fold'], 'nn_category.joblib'))
    #rfc_category = load(os.path.join(args.path['out_fold'], 'rfc_category.joblib'))
    #svc_category = load(os.path.join(args.path['out_fold'], 'svc_category.joblib'))

    # Load the test data for sklearn models
    #_, _, binary_test_x, binary_test_y = load_baseline_date(args.path['out_fold'], 'binary')
    _, _, category_test_x, category_test_y = load_baseline_date(args.path['out_fold'], 'category')

    # Load the trained bert models
    #binary_model, binary_loader = get_bert_model_and_data(conf)
    category_model, category_loader = get_bert_model_and_data(conf)

    #roc = ROC(task="category")
    #y_preds_binary, y_label_binary = evaluation(args, binary_model, binary_loader)
    y_preds_category, y_label_category = evaluation(args, category_model, category_loader)
    evaluator = Evaluator(conf=conf, device=args.device)
    #metrics = evaluator.calculate_metrics(y_preds_binary, y_label_binary)
    metrics = evaluator.calculate_metrics(y_preds_category, y_label_category)
    print(metrics)
    exit()
    th_fpr, th_tpr, thresholds = roc(y_preds, y_label)



    # Extract fpr and tpr for each example
    rfc_proba = rfc_binary.predict_proba(binary_test_x)[::, 1]
    rfc_fpr, rfc_tpr, _ = metrics.roc_curve(binary_test_y, rfc_proba)
    nn_proba = nn_binary.predict_proba(binary_test_x)[::, 1]
    nn_fpr, nn_tpr, _ = metrics.roc_curve(binary_test_y, nn_proba)
    svc_proba = svc_binary.predict_proba(binary_test_x)[::, 1]
    svc_fpr, svc_tpr, _ = metrics.roc_curve(binary_test_y, svc_proba)

    # create ROC curve
    plt.plot(rfc_fpr, rfc_tpr)
    plt.plot(nn_fpr, nn_tpr)
    plt.plot(svc_fpr, svc_tpr)
    plt.plot(th_fpr.cpu().numpy(), th_tpr.cpu().numpy())
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    plt.savefig(f'{args.path["out_fold"]}/roc_binary.png')
