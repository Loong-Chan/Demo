import torch
from torch import nn, optim
import torch.nn.functional as F
from pytorch_lightning import seed_everything
import gc

import Utils as U
from TrainingLogger import MultipleTrialLogger
from Dataset import InMemoryNodeDataset
from Model import GCN


def train_step(model: nn.Module, 
               input_dict: dict,
               labels,
               optimizer, 
               train_idx=None):
    model.train()
    output = model(**input_dict)
    if train_idx is None:
        train_loss = F.nll_loss(output, labels)
    else:
        train_loss = F.nll_loss(output[train_idx], labels[train_idx])
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
    return [train_loss]


def val_step(model: nn.Module,
             input_dict: dict,
             labels,
             val_idx=None):
    model.eval()
    with torch.no_grad():
        output = model(**input_dict)
        if val_idx is None:
            val_loss = F.nll_loss(output, labels)
        else:
            val_loss = F.nll_loss(output[val_idx], labels[val_idx])
    return [val_loss]


def test_step(model: nn.Module,
             input_dict: dict,
             labels,
             test_idx=None):
    model.eval()
    with torch.no_grad():
        output = model(**input_dict)
        if test_idx is None:
            test_acc = U.accuary(output, labels)
        else:
            test_acc = U.accuary(output[test_idx], labels[test_idx])
    return [test_acc]


def main(args):
    data = InMemoryNodeDataset(args).to(args.device)
    x = data.x
    edge_index = data.edge_index
    y = data.y

    logger = MultipleTrialLogger()
    for trial in range(args.num_trial):
        logger.new_trial(args)
        split = data.random_split(seed=trial, 
                                  p_train=args.proportion_train, 
                                  p_val=args.proportion_val, 
                                  device=args.device)
        train_idx, val_idx, test_idx = split
        seed_everything(trial)
        model = GCN(args).to(args.device)
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.optimizer_lr, 
            weight_decay=args.optimizer_weight_decay)
        # 一般常有的有两个：ReduceLROnPlateau 和 CosineAnnealingWarmRestarts
        schedular = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=args.schedular_factor,
            patience=args.schedular_patience,
            threshold=args.schedular_threshold)

        for epoch in range(args.max_epoch):
            
            # Step 1: Train
            output = train_step(model=model,
                                input_dict={"x": x, "edge_index": edge_index},
                                labels=y,
                                optimizer=optimizer,
                                train_idx=train_idx)
            train_loss = output[0]

            # Step 2: Validation
            output = val_step(model=model,
                              input_dict={"x": x, "edge_index": edge_index},
                              labels=y,
                              val_idx=val_idx)
            val_loss = output[0]

            # Step 3: Test
            output = test_step(model=model,
                               input_dict={"x": x, "edge_index": edge_index},
                               labels=y,
                               test_idx=test_idx)
            test_acc = output[0]
            gc.collect()

            # Step 4: Modify learning rate (optional)
            schedular.step(val_loss)

            # Step 5: Log info
            logger.log(epoch=epoch, train_loss=train_loss.item(), 
                       val_loss=val_loss.item(), test_acc=test_acc.item())
            if args.verbose:
                print(f"[Epoch {epoch:4d} LR {schedular._last_lr[0]:.4f}]",
                      f" Train Loss: {train_loss:.4f} Valid Loss: {val_loss:.4f}",
                      f", Test Acc: {test_acc*100:.2f}%.", flush=True)

            # Step 6: early stop (optional)
            if logger.early_stop():
                break

        if args.verbose:
            print(f"Test accuracy with seed {trial}: {logger.best_test_acc()*100:.2f}%.", flush=True)

    mean, std = logger.result()
    print(f"The results of {args.num_trial} trial(s): Mean: {mean*100:.2f}%, Std: {std*100:.2f}%.", flush=True)
    return mean


if __name__ == "__main__":
    args = U.load_args_from_commands()
    main(args)
