from transformermodule.Trainer import Trainer
from tqdm import tqdm
import torch as th
import time


class LOSTrainer(Trainer):
    def __init__(self, args):
        super(LOSTrainer, self).__init__(args)

    def train(self):

        for e in range(0, self.epochs):
            train_loss, train_metrics = self.evaluation(self.train_loader)
            eval_loss, eval_metrics = self.evaluation(self.evalu_loader)
            train_metrics.update({'loss': train_loss})
            eval_metrics.update({'loss': eval_loss})

            self.logger.log_metrics(train_metrics, 'train')
            self.logger.log_metrics(eval_metrics, 'eval')

            self.logger.report_metrics(train_metrics, 'train')
            self.logger.report_metrics(eval_metrics, 'eval')

            if self.stopper.step(eval_loss, self.model):
                self.logger.info('Early Stop!\tEpoch:' + str(e))
                break

            self.epoch(e)

        print('Evaluating trained model')
        self.model = self.stopper.load_model(self.model)

        test_loss, test_metrics = self.evaluation(self.test_loader)
        test_metrics.update({'loss': test_loss})

        self.logger.report_metrics(test_metrics, 'test')
        self.logger.log_value('test_loss', test_loss)
        self.logger.log_values(test_metrics, 'test')

        self.logger.stop_log()

    def epoch(self, e):
        self.model.train()
        tr_loss, step = 0, 0
        epoch_time = time.time()

        loader_iter = tqdm(self.train_loader, ncols=120, position=0)
        for step, batch in enumerate(loader_iter, 1):
            batch = tuple(t.to(self.args.device) for t in batch)
            input_ids, posi_ids, age_ids, gender_ids, att_mask, labels, pat_ids = batch

            loss, logits = self.model(
                input_ids=input_ids,
                posi_ids=posi_ids,
                age_ids=age_ids,
                gender_ids=gender_ids,
                targets=labels,
                attention_mask=att_mask
            )

            loss.backward()

            tr_loss += loss.item()
            print_dict = {'epoch': e, 'loss': tr_loss / step}
            loader_iter.set_postfix(print_dict)

            self.optim.step()
            self.optim.zero_grad()

        return tr_loss / step, time.time() - epoch_time

    def evaluation(self, loader):
        self.model.eval()
        tr_loss, step = 0, 0
        y_preds, y_label = None, None
        for step, batch in enumerate(loader, 1):
            batch = tuple(t.to(self.args.device) for t in batch)
            input_ids, posi_ids, age_ids, gender_ids, att_mask, labels, pat_ids = batch

            with th.no_grad():
                loss, logits = self.model(
                    input_ids=input_ids,
                    posi_ids=posi_ids,
                    age_ids=age_ids,
                    gender_ids=gender_ids,
                    targets=labels,
                    attention_mask=att_mask
                )

            tr_loss += loss.item()

            if self.task in ['binary', 'category']:
                y_preds = logits if y_preds is None else th.cat((y_preds, logits))
                y_label = labels if y_label is None else th.cat((y_label, labels))
            elif self.task == 'real':
                y_preds = [logits] if y_preds is None else y_preds + [logits]
                y_label = [labels] if y_label is None else y_label + [labels]

        metrics = self.evaluator.calculate_metrics(y_preds, y_label)

        return tr_loss / step, metrics
