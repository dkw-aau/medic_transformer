from multiprocessing import freeze_support
from torch.utils.data import DataLoader

from Utils.EarlyStopping import EarlyStopping
from transformermodule.Evaluator import Evaluator
from .Model.MBERT import MBERT
from .utils import get_model_config
from .DataLoader.MLMLoader import MLMLoader
from .Model.utils import BertConfig
from .Model.optimiser import adam
from Utils.utils import load_corpus, create_folder
from tqdm import tqdm
import warnings
import os


class MLMTrainer:
    def __init__(
            self,
             args):

        self.args = args
        freeze_support()
        self.args = args
        self.logger = self.args.logger
        warnings.filterwarnings(action='ignore')
        create_folder(args.path['out_fold'])

        # TODO: Implement AUC-ROC, AUC-PR, KAPPA, print confusion matrix
        self.conf = {
            'task': 'mlm',
            'metrics': ['auc', 'f1'],
            'binary_thresh': 2,
            'cats': [2, 7],
            'years': [2018, 2019, 2020, 2021],
            'types': ['apriori', 'adm', 'proc', 'vital', 'lab'],
            # 'apriori', 'vital', 'diag', 'apriori', 'adm', 'proc', 'lab'
            'max_hours': 24
        }

        # Load corpus
        corpus = load_corpus(os.path.join(args.path['data_fold'], args.corpus_name))

        vocab = corpus.prepare_corpus(
            self.conf
        )
        corpus.create_pos_ids(event_dist=300)
        corpus.create_train_evel_test_idx(train_size=0.8)

        # Select subset corpus
        train_x, _, _ = corpus.split_train_eval_test()

        # Setup Evaluator
        self.evaluator = Evaluator(conf=self.conf, device=self.args.device)

        # Setup dataloader
        Dset = MLMLoader(
            token2idx=vocab['token2index'],
            sequences=train_x,
            max_len=args.max_len_seq)

        self.trainload = DataLoader(dataset=Dset, batch_size=args.batch_size, shuffle=True, num_workers=0)

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
        model = MBERT(
            args=args,
            bert_conf=bert_conf,
            feature_dict=feature_dict,
            cls_conf=self.conf)

        self.model = model.to(args.device)

        self.optim = adam(params=list(self.model.named_parameters()), args=args)

    def train(self, epochs):

        # Start Logger
        self.logger.start_log()
        self.logger.log_value('Experiment', self.args.experiment_name)

        # Create earlystopper
        stopper = EarlyStopping(
            patience=self.args.patience,
            save_path=f'{os.path.join(self.args.path["out_fold"], self.conf["task"])}.pt',
            save_trained=self.args.save_model
        )

        for e in range(0, epochs):
            e_loss = self.epoch()

            # Log loss
            self.logger.log_metrics({'loss': e_loss}, 'train')

            print(f'Loss: {e_loss}')
            if stopper.step(e_loss, self.model):
                self.logger.info('Early Stop!\tEpoch:' + str(e))
                break

    def epoch(self):
        tr_loss = 0

        loader_iter = tqdm(self.trainload)
        for step, batch in enumerate(loader_iter, 1):
            batch = tuple(t.to(self.args.device) for t in batch)
            input_ids, posi_ids, age_ids, gender_ids, attMask, label, idx = batch

            loss, pred, label = self.model(
                input_ids=input_ids,
                posi_ids=posi_ids,
                age_ids=age_ids,
                gender_ids=gender_ids,
                attention_mask=attMask,
                targets=label
            )

            loss.backward()

            tmp_loss = loss.item()
            tr_loss += tmp_loss

            self.optim.step()
            self.optim.zero_grad()

        return tr_loss / step
