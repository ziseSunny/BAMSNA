import torch
import torch.nn.functional as F
import torch_geometric
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv
from directednet import DirectedUnweighted
from directedsage import DirectedSAGE
from directedgat import DirectedGAT
from directedgcn import DirectedGCN
from sagedirected import SAGEDirected
from gatdirected import GATDirected
from directedgin import DirectedGIN
from gindirected import GINDirected

import math

class LGCNDirected(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size=512, K=2):
        super(LGCNDirected, self).__init__()
        self.conv1 = DirectedUnweighted(K=K)
        self.linear = torch.nn.Linear(input_size * 2 * (K + 1), output_size)
    def forward(self, feature, edge_index):
        x = self.conv1(feature, edge_index)
        x = self.linear(x)
        return x

class LSAGEDirected(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size=512, K=2):
        super(LSAGEDirected, self).__init__()
        self.conv1 = DirectedSAGE(K=K)
        self.linear = torch.nn.Linear(input_size * pow(2, (K + 1)), output_size)
    def forward(self, feature, edge_index):
        x = self.conv1(feature, edge_index)
        x = self.linear(x)
        return x

class LGATDirected(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size=512, K=2, dropout=0.5):
        super(LGATDirected, self).__init__()
        self.conv1 = DirectedGAT(in_channels=input_size, K=K, heads=1, concat=False, dropout=dropout)
        self.linear = torch.nn.Linear(input_size * 2 * (K + 1), output_size)
    def forward(self, feature, edge_index):
        x = self.conv1(feature, edge_index)
        x = self.linear(x)
        return x

class LGINDirected(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size=512, K=2, dropout=0.5):
        super(LGINDirected, self).__init__()
        self.conv1 = DirectedGIN(K=K, train_eps=False)
        self.linear = torch.nn.Sequential(torch.nn.Linear(input_size * 2 * (K + 1), hidden_size), torch.nn.ReLU(),
                                          torch.nn.Linear(hidden_size, output_size))
    def reset_parameters(self):
        init_weight(self.linear)

    def forward(self, feature, edge_index):
        x = self.conv1(feature, edge_index)
        x = self.linear(x)
        return x

class GCNNetDirected(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size=512):
        super(GCNNetDirected, self).__init__()
        self.conv1 = DirectedGCN()
        self.linear1 = torch.nn.Linear(input_size * 2, hidden_size * 2)
        self.conv2 = DirectedGCN()
        self.linear2 = torch.nn.Linear(hidden_size * 2, output_size)
    def forward(self, feature, edge_index):
        x = F.dropout(feature, p=0.5, training=self.training)
        x = F.elu(self.conv1(torch.cat([x,x],dim=1), edge_index))
        x = self.linear1(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.linear2(x)
        return x

class SAGENetDirected(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size=512):
        super(SAGENetDirected, self).__init__()
        self.conv1 = SAGEDirected()
        self.linear1 = torch.nn.Linear(input_size * 4, hidden_size * 2)
        self.conv2 = SAGEDirected()
        self.linear2 = torch.nn.Linear(hidden_size * 4, output_size)
    def forward(self, feature, edge_index):
        x = F.dropout(feature, p=0.5, training=self.training)
        x = F.elu(self.conv1(torch.cat([x,x], dim=1), edge_index))
        x = self.linear1(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.linear2(x)
        return x

class GATNetDirected(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size=512, dropout=0.5):
        super(GATNetDirected, self).__init__()
        self.conv1 = GATDirected(in_channels=input_size, heads=1, concat=False, dropout=dropout)
        self.linear1 = torch.nn.Linear(input_size * 2, hidden_size * 2)
        self.conv2 = GATDirected(in_channels=hidden_size, heads=1, concat=False, dropout=dropout)
        self.linear2 = torch.nn.Linear(hidden_size * 2, output_size)
    def forward(self, feature, edge_index):
        x = F.dropout(feature, p=0.5, training=self.training)
        x = F.elu(self.conv1(torch.cat([x,x], dim=1), edge_index))
        x = self.linear1(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.linear2(x)
        return x

class GINNetDirected(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size=512):
        super(GINNetDirected, self).__init__()
        self.conv1 = GINDirected()
        self.linear1 = torch.nn.Sequential(torch.nn.Linear(input_size * 2, hidden_size), torch.nn.ReLU(),
                                           torch.nn.Linear(hidden_size, hidden_size * 2))
        self.conv2 = GINDirected()
        self.linear2 = torch.nn.Sequential(torch.nn.Linear(hidden_size * 2, hidden_size), torch.nn.ReLU(),
                                           torch.nn.Linear(hidden_size, output_size))
    def reset_parameters(self):
        init_weight(self.linear1)
        init_weight(self.linear2)

    def forward(self, feature, edge_index):
        x = F.dropout(feature, p=0.5, training=self.training)
        x = F.elu(self.conv1(torch.cat([x,x], dim=1), edge_index))
        x = self.linear1(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.linear2(x)
        return x

class GCNNet(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size=512):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(input_size, hidden_size)
        self.conv2 = GCNConv(hidden_size, output_size)
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
    def forward(self, feature, edge_index):
        x = F.dropout(feature, p=0.5, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class GATNet(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size=512, heads=1):
        super(GATNet, self).__init__()
        self.conv1 = GATConv(input_size, hidden_size, heads=heads) #可以多头机制
        self.conv2 = GATConv(hidden_size * heads, output_size)
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
    def forward(self, feature, edge_index):
        x = F.dropout(feature, p=0.5, training=self.training)
        x = F.elu(self.conv1(x, edge_index)) #将elu作为激活函数，alpha的默认值为1.0
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class SAGENet(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size=512):
        super(SAGENet, self).__init__()
        self.conv1 = SAGEConv(input_size, hidden_size)
        self.conv2 = SAGEConv(hidden_size, output_size)
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
    def forward(self, feature, edge_index):
        x = F.dropout(feature, p=0.5, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GINNet(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size=512):
        super(GINNet, self).__init__()
        self.conv1 = GINConv(torch.nn.Sequential(torch.nn.Linear(input_size, hidden_size), torch.nn.ReLU(),
                                                 torch.nn.Linear(hidden_size, hidden_size)), train_eps=False)
        self.conv2 = GINConv(torch.nn.Sequential(torch.nn.Linear(hidden_size, hidden_size), torch.nn.ReLU(),
                                                 torch.nn.Linear(hidden_size, output_size)), train_eps=False)
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        init_weight(self.conv1)
        init_weight(self.conv2)
    def forward(self, feature, edge_index):
        x = F.dropout(feature, p=0.5, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class WDiscriminator(torch.nn.Module):
    def __init__(self, hidden_size, hidden_size2=512):
        super(WDiscriminator, self).__init__()
        self.hidden = torch.nn.Linear(hidden_size, hidden_size2)
        self.hidden2 = torch.nn.Linear(hidden_size2, hidden_size2)
        self.output = torch.nn.Linear(hidden_size2, 1)
    def forward(self, input_embd):
        return self.output(F.leaky_relu(self.hidden2(F.leaky_relu(self.hidden(input_embd), 0.2, inplace=True)), 0.2, inplace=True))

class transformation(torch.nn.Module):
    def __init__(self, hidden_size=512, hidden_size2=512):
        super(transformation, self).__init__()
        self.trans = torch.nn.Parameter(torch.eye(hidden_size))
    def forward(self, input_embd):
        return input_embd.mm(self.trans)

class notrans(torch.nn.Module):
    def __init__(self):
        super(notrans, self).__init__()
    def forward(self, input_embd):
        return input_embd

class ReconDNN(torch.nn.Module):
    def __init__(self, hidden_size, feature_size, hidden_size2=512):
        super(ReconDNN, self).__init__()
        self.hidden = torch.nn.Linear(hidden_size, hidden_size2)
        self.output = torch.nn.Linear(hidden_size2, feature_size)
    def forward(self, input_embd):
        return self.output(F.relu(self.hidden(input_embd)))