import torch


class Logger:
    def __init__(self, args):
        self.early_stop_patience = args.logger_early_stop_patience
        self.best_epoch = -1
        self.current_epoch = -1
        self.min_val_loss = torch.inf
        self.best_test_acc = 0.
        self.records = []

    def log(self, **kwargs):
        self.records.append(kwargs)
        self.current_epoch = self.current_epoch + 1
        if self.min_val_loss > kwargs["val_loss"]:
            self.best_epoch = self.current_epoch
            self.min_val_loss = kwargs["val_loss"]
            self.best_test_acc = kwargs["test_acc"]

    def early_stop(self):
        return self.current_epoch - self.best_epoch > self.early_stop_patience


class MultipleTrialLogger:
    def __init__(self):
        self.logger_list = []

    def new_trial(self, args):
        self.logger_list.append(Logger(args))

    def log(self, **kwargs):
        self.logger_list[-1].log(**kwargs)

    def early_stop(self):
        return self.logger_list[-1].early_stop()

    def best_test_acc(self):
        return self.logger_list[-1].best_test_acc

    def result(self):
        trial_results = [logger.best_test_acc for logger in self.logger_list]
        trial_results = torch.tensor(trial_results)
        return trial_results.mean(), trial_results.std()
