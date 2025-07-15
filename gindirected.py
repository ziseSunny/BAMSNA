import torch
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops

class GINDirected(MessagePassing):
    def __init__(self, eps=0, train_eps=False, **kwargs):
        super(GINDirected, self).__init__(aggr='add', **kwargs)
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x, edge_index):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_index, _ = remove_self_loops(edge_index)
        edge_index_t = torch.vstack([edge_index[1, :], edge_index[0, :]])

        xs_r = (1 + self.eps) * x[:, :int(x.size(1)/2)] + self.propagate(edge_index, x=x[:, :int(x.size(1)/2)])
        xs_t = (1 + self.eps) * x[:, int(x.size(1)/2):] + self.propagate(edge_index_t, x=x[:, int(x.size(1)/2):])

        out = torch.cat([xs_r, xs_t], dim=1)

        # out = self.nn((1 + self.eps) * x + self.propagate(edge_index, x=x))
        return out

    def message(self, x_j):
        return x_j

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)