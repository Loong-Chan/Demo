from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from Model.ModelUtils import *


class GCN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dropout = args.dropout
        self.num_layer = args.num_layer

        input_dims = [args.num_feat] + args.num_layer * [args.num_hidden]
        output_dims = args.num_layer * [args.num_hidden] + [args.num_class]
        self.convs = nn.ModuleList()
        for in_, out_ in zip(input_dims, output_dims):
            self.convs.append(GCNConv(in_ ,out_))

    def forward(self, x, edge_index):        
        for idx, conv in enumerate(self.convs):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)
            if idx < self.num_layer - 1:
                x = F.relu(x)
            else:
                x = F.log_softmax(x, dim=1)
        return x
