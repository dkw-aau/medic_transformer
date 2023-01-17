import copy

from Utils.utils import save_model_state, load_state_dict


class EarlyStopping(object):
    def __init__(self, patience=10, save_path=None, save_trained=True):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.save_trained = save_trained
        if save_path is None:
            self.best_model = None
        self.save_path = save_path

    def step(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_model(model)
        elif score >= self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.save_model(model)
            self.best_score = score
            self.counter = 0
        return self.early_stop

    def save_model(self, model):
        if self.save_trained:
            model.eval()
            save_model_state(model, self.save_path)
        else:
            self.best_model = copy.deepcopy(model)

    def load_model(self, model):
        if self.save_trained:
            return load_state_dict(self.save_path, model)
        else:
            return self.best_model
