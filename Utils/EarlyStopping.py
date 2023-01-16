import copy

from Utils.utils import save_model_state, load_state_dict


class EarlyStopping(object):
    def __init__(self, patience=10, save_path=None):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
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
        if self.save_path is None:
            self.best_model = copy.deepcopy(model)
        else:
            model.eval()
            save_model_state(model, self.save_path)

    def load_model(self, model):
        if self.save_path is None:
            return self.best_model
        else:
            return load_state_dict(self.save_path, model)
