import random
import numpy as np
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import remove_self_loops, to_undirected

import Utils as U
import Dataset.DatasetUtils as DU


class InMemoryNodeDataset:
    def __init__(self, args):
        self.dataset = args.dataset
        self.file_root = DU.dateset_root()
        if self.dataset in ['cora', 'citeseer']:
            data = Planetoid(root=self.file_root, name=self.dataset)
        else:
            raise NotImplementedError

        self.x = data.x
        self.edge_index = data.edge_index
        self.y = data.y

        self.num_feat = self.x.shape[1]
        self.num_class = self.y.max().item() + 1
        self.num_node = self.x.shape[0]
        U.set_attr(args, "num_feat", self.num_feat)
        U.set_attr(args, "num_class", self.num_class)
        U.set_attr(args, "num_node", self.num_node)

        if args.dataset_remove_selfloop == True:
            self.edge_index = remove_self_loops(self.edge_index)[0]
        if args.dataset_undirected == True:
            self.edge_index = to_undirected(self.edge_index)
        if args.dataset_row_normalize == True:
            self.x[self.x.isnan()] = 0.
            rowsum = self.x.sum(dim=1, keepdim=True)
            rowsum[rowsum == 0.] = 1.
            self.x = self.x / rowsum

    def to(self, device):
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        self.y = self.y.to(device)
        return self

    def random_split(self, seed, p_train, p_val, device=None):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        num_train = int(self.num_node * p_train)
        num_val = int(self.num_node * p_val)
        full_idx = torch.randperm(self.num_node)
        train_idx = full_idx[:num_train]
        val_idx = full_idx[num_train+1:num_train+num_val]
        test_idx = full_idx[num_train+num_val+1:]
        if device is not None:
            train_idx = train_idx.to(device)
            val_idx = val_idx.to(device)
            test_idx = test_idx.to(device)
        return train_idx, val_idx, test_idx
