import math
import os

import matplotlib.pyplot as plt
import seaborn as sns
import torch as th
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from torchmetrics import ROC

from Utils.Evaluator import Evaluator
from Utils.utils import set_seeds, load_state_dict
from baselinemodule.BaselineDataset import BaselineDataset
from config import Config
from transformermodule.Trainer import Trainer


class Plots(Trainer):
    def __init__(self, args):

        args.task = 'binary'
        plot_ext = 'eps'

        print("Creating corpus")
        super(Plots, self).__init__(args)

        print('Preloading model')
        self.model = load_state_dict(os.path.join(self.path['out_fold'], 'binary.pt'), self.model)

        print('Creating baseline datasets')
        self.dataset = BaselineDataset(self.corpus, self.corpus_conf)

        # Load the test data for sklearn models
        train_x, train_y = self.dataset.train_x_bin, self.dataset.train_y_bin
        test_x, test_y = self.dataset.test_x_bin, self.dataset.test_y_bin

        # Train baseline models
        print(f'\nTraining Baseline Classifiers')
        rfc_model = RandomForestClassifier(random_state=42)
        ann_model = MLPClassifier(random_state=42)
        svm_model = SVC(probability=True)
        rfc_model.fit(train_x, train_y)
        ann_model.fit(train_x, train_y)
        svm_model.fit(train_x, train_y)

        # Create evaluator
        evaluator = Evaluator(task=args.task, conf=self.corpus_conf)

        # Create prediction chunks
        age_rfc_fprs = {}
        gender_rfc_fprs = {}
        age_aucs = {}
        gender_aucs = {}
        age_chunks, gender_chunks = self.evaluate_chunks(args, self.model, self.test_loader)
        age_chunks = dict(sorted(age_chunks.items()))
        for name, chunk in age_chunks.items():
            if len(chunk[0]) > 100:
                age_rfc_fprs[name] = self.get_fpr_tpr(chunk[0], chunk[1], tensor=True)
                age_aucs[name] = evaluator.calculate_metrics(chunk[0], chunk[1])['auc']

        for name, chunk in gender_chunks.items():
            gender_rfc_fprs[name] = self.get_fpr_tpr(chunk[0], chunk[1], tensor=True)
            gender_aucs[name] = evaluator.calculate_metrics(chunk[0], chunk[1])['auc']

        # Create probabilities
        th_proba, th_test_y = self.evaluation(args, self.model, self.test_loader)
        rfc_proba = rfc_model.predict_proba(test_x)[:, 1]
        nn_proba = ann_model.predict_proba(test_x)[:, 1]
        svc_proba = svm_model.predict_proba(test_x)[:, 1]

        # Get model metrics
        th_auc = evaluator.calculate_metrics(th_proba, th_test_y)['auc']
        rfc_auc = roc_auc_score(test_y, rfc_proba, multi_class='ovo')
        nn_auc = roc_auc_score(test_y, nn_proba, multi_class='ovo')
        svc_auc = roc_auc_score(test_y, svc_proba, multi_class='ovo')

        # Extract fpr and tpr for each example
        th_fpr, th_tpr = self.get_fpr_tpr(th_proba, th_test_y, tensor=True)
        rfc_fpr, rfc_tpr = self.get_fpr_tpr(rfc_proba, test_y)
        nn_fpr, nn_tpr = self.get_fpr_tpr(nn_proba, test_y)
        svc_fpr, svc_tpr = self.get_fpr_tpr(svc_proba, test_y)

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

        plt.savefig(f'{args.path["out_fold"]}/roc_{self.task}.{plot_ext}', format=plot_ext)
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

        plt.savefig(f'{args.path["out_fold"]}/roc_{self.task}_ages.{plot_ext}', format=plot_ext)
        plt.clf()

        # Create ROC curve for genders
        for rfc_fpr, gender in zip(gender_rfc_fprs.items(), gender_aucs.items()):
            sex = 'Male' if int(gender[0]) == 0 else 'Female'

            plt.plot(rfc_fpr[1][0], rfc_fpr[1][1], label=f'AUROC {sex} = {round(gender[1], 2)}')

        plt.plot([0, 1], [0, 1], color='orange', linestyle='--')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.legend(loc='lower right')

        plt.savefig(f'{args.path["out_fold"]}/roc_{self.task}_gender.{plot_ext}', format=plot_ext)
        plt.clf()

    def get_fpr_tpr(self, proba, labs, tensor=False):
        if tensor:
            roc = ROC(task='binary', num_classes=2)
            fpr, tpr, _ = roc(proba, labs)
            fpr = fpr.cpu().numpy()
            tpr = tpr.cpu().numpy()
        else:
            fpr, tpr, _ = metrics.roc_curve(labs, proba)

        return fpr, tpr

    def evaluation(self, args, model, loader):
        model.eval()
        y_preds, y_label = None, None
        for step, batch in enumerate(loader, 1):
            batch = tuple(t.to(args.device) for t in batch)
            input_ids, posi_ids, age_ids, gender_ids, att_mask, labels, pat_ids = batch

            with th.no_grad():
                loss, logits = model(input_ids, posi_ids, age_ids, gender_ids, targets=labels, attention_mask=att_mask)

            y_preds = logits if y_preds is None else th.cat((y_preds, logits))
            y_label = labels if y_label is None else th.cat((y_label, labels))

        return y_preds, y_label

    def evaluate_chunks(self, args, model, loader):
        model.eval()
        age_chunk_dict = {}
        gender_chunk_dict = {}
        for step, batch in enumerate(loader, 1):
            batch = tuple(t.to(args.device) for t in batch)
            input_ids, posi_ids, age_ids, gender_ids, att_mask, labels, pat_ids = batch

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

    Plots(args)


