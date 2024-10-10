import sys
import copy
import torch
import argparse
from pprint import pprint


def test_attr(
        args: argparse.Namespace, 
        attr_name: str, 
        test_value
    ) -> bool:
    if not isinstance(args, argparse.Namespace):
        raise TypeError(f"Expected 'args' to be an instance of 'argparse.Namespace',",
                        f"but got {type(args).__name__}.")
    if not isinstance(attr_name, str):
        raise TypeError(f"Expected 'attr_name' to be an instance of 'str',",
                        f"but got {type(attr_name).__name__}.")
    if not hasattr(args, attr_name):
        raise ValueError(f"[Error] args do not have attribute \"{attr_name}\".")
    return getattr(args, attr_name) == test_value


def set_attr(
        args:argparse.Namespace,
        attr_name: str,
        new_value,
        verbose: bool=True
    ) -> None:
    if not isinstance(args, argparse.Namespace):
        raise TypeError(f"Expected 'args' to be an instance of 'argparse.Namespace',", 
                        f"but got {type(args).__name__}.")
    if not isinstance(attr_name, str):
        raise TypeError(f"Expected 'attr_name' to be an instance of 'str',",
                        f"but got {type(attr_name).__name__}.")
    
    if hasattr(args, attr_name):
        current_value = getattr(args, attr_name)
        expected_type = type(current_value)
        if not isinstance(new_value, expected_type):
            raise TypeError(f"Expected new value for 'args.{attr_name}' to be of type {expected_type.__name__}, but got {type(new_value).__name__}.")
    else:
        expected_type = None

    if verbose:
        if expected_type is None:
            print(f"[INFO] Set new attribute args.{attr_name} = {new_value}.", flush=True)
        else:
            print(f"[INFO] Modify args.{attr_name} from {current_value} to {new_value}.", flush=True)
   
    setattr(args, attr_name, new_value)

def merge_args(
        incomplete_args: argparse.Namespace, 
        full_args: argparse.Namespace, 
        verbose: bool=True
    ) -> None:
    if not isinstance(incomplete_args, argparse.Namespace):
        raise TypeError(f"Expected 'incomplete_args' to be an instance of 'argparse.Namespace', but got {type(incomplete_args).__name__}.")
    for key, value in vars(full_args).items():
        if not hasattr(incomplete_args, key):
            set_attr(incomplete_args, key, value, verbose=verbose)


def print_args(
        args: argparse.Namespace
    ) -> None:
    pprint("Default Params:")
    pprint(vars(args))
    sys.stdout.flush()


def load_args_from_commands() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    command_args, unknown_args = parser.parse_known_args()

    unknown_args_dict = {}
    try:
        for i in range(0, len(unknown_args), 2):
            key = unknown_args[i].lstrip('-')
            value = unknown_args[i+1] if i+1 < len(unknown_args) else None
            unknown_args_dict[key] = value
    except IndexError:
        raise ValueError("Incorrect number of command line arguments provided.")

    for key, value in unknown_args_dict.items():
        set_attr(command_args, key, value, verbose=False)

    sys.argv = [sys.argv[0]]

    from DefaultArgs import default_args
    args = copy.deepcopy(default_args)
    for key, value in vars(command_args).items():
        if not hasattr(args, key):
            raise KeyError(f"Argument '{key}' is not a recognized default argument.")
        else:
            default_type = type(getattr(args, key))
            try:
                value = default_type(value)
            except ValueError:
                raise ValueError(f"Cannot convert argument '{key}' to {default_type}.")
            set_attr(args, key, value, verbose=False)

    print_args(args)
    return args


def accuary(predict, target):
    _, predicted_classes = torch.max(predict, 1)
    correct = (predicted_classes == target).sum()
    total = target.size(0)
    accuracy = correct / total
    return accuracy
