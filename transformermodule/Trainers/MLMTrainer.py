from tqdm import tqdm

from transformermodule.Trainer import Trainer


class MLMTrainer(Trainer):
    def __init__(self, args):
        super(MLMTrainer, self).__init__(args)

    def train(self):
        for e in range(0, self.epochs):
            e_loss = self.epoch()

            # Log loss
            self.logger.log_metrics({'loss': e_loss}, 'train')

            print(f'Loss: {e_loss}')
            if self.stopper.step(e_loss, self.model):
                self.logger.info('Early Stop!\tEpoch:' + str(e))
                break

    def epoch(self):
        tr_loss, step = 0, 0

        loader_iter = tqdm(self.train_loader)
        for step, batch in enumerate(loader_iter, 1):
            batch = tuple(t.to(self.args.device) for t in batch)
            input_ids, posi_ids, age_ids, gender_ids, att_mask, labels, idx = batch

            loss = self.model(
                input_ids=input_ids,
                posi_ids=posi_ids,
                age_ids=age_ids,
                gender_ids=gender_ids,
                attention_mask=att_mask,
                targets=labels
            )

            loss.backward()

            tmp_loss = loss.item()
            tr_loss += tmp_loss

            self.optim.step()
            self.optim.zero_grad()

        return tr_loss / step
