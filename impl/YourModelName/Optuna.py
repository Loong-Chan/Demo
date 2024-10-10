import sys
import optuna
import argparse
import copy
import json

from Main import main
import Utils as U


class Objective:
    def __init__(self, full_args):
        self.full_args = copy.deepcopy(full_args)

    def __call__(self, trial):
        trial_args = argparse.Namespace()
        #在这里设置搜参空间#
        trial_args.optimizer_lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        trial_args.optimizer_weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        trial_args.dropout = trial.suggest_float("dropout", 0.01, 0.99, log=False)
        trial_args.num_hidden = trial.suggest_categorical("num_hidden", [64, 128, 256])
        trial_args.num_layer = trial.suggest_int("num_layer", 2, 5) 
        #################
        U.merge_args(trial_args, self.full_args, verbose=False)
        trial.set_user_attr("args", vars(trial_args).copy())
        return main(trial_args)


def save_best_args_callback(study, trial):
    if trial.number == study.best_trial.number:
        dataset = trial.user_attrs["args"]["dataset"]
        with open("best_args.json", 'r') as f:
            args_dict = json.load(f)
            args_dict[dataset] = trial.user_attrs["args"]
        with open("best_args.json", 'w') as f:
            json.dump(args_dict, f, indent=4)


if __name__ == "__main__":
    args = U.load_args_from_commands()
    U.set_attr(args, "verbose", False)
    study = optuna.create_study(direction='maximize')
    objective = Objective(args)
    study.optimize(objective, 
                   n_trials=args.num_try, 
                   callbacks=[save_best_args_callback])
