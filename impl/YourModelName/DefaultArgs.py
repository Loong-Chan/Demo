import argparse


parser = argparse.ArgumentParser()

# Command Parameters
parser.add_argument("--dataset", default="cora", type=str)
parser.add_argument("--device", default="cuda:0", type=str)
parser.add_argument("--verbose", default=True, type=bool)

# Dataset Preprocessing Parameters
parser.add_argument("--dataset_remove_selfloop", default=True, type=bool)
parser.add_argument("--dataset_undirected", default=True, type=bool)
parser.add_argument("--dataset_row_normalize", default=True, type=bool)

# Fine-tune Setting
parser.add_argument("--num_try", default=300, type=int)

# Training Setting
parser.add_argument("--num_trial", default=10, type=int)
parser.add_argument("--max_epoch", default=2000, type=int)
parser.add_argument("--optimizer_lr", default=0.01, type=float)  # Adam optimizer by default
parser.add_argument("--optimizer_weight_decay", default=5e-4, type=float)
parser.add_argument("--schedular_factor", default=0.5, type=float)  # ReduceLROnPlateau schedular by default
parser.add_argument("--schedular_patience", default=50, type=int)
parser.add_argument("--schedular_threshold", default=1e-5, type=float)
parser.add_argument("--logger_early_stop_patience", default=200, type=int)
parser.add_argument("--proportion_train", default=0.6, type=float)
parser.add_argument("--proportion_val", default=0.2, type=float)

# Network Setting
parser.add_argument("--num_hidden", default=64, type=int)
parser.add_argument("--num_layer", default=2, type=int)
parser.add_argument("--dropout", default=0.5, type=float)

default_args = parser.parse_args()
