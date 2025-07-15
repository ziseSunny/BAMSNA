import torch
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops

class DirectedGIN(MessagePassing):
    def __init__(self, K, eps=0, train_eps=False, **kwargs):
        super(DirectedGIN, self).__init__(aggr='add', **kwargs)
        self.initial_eps = eps
        self.K = K
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

        xs_r = [x]
        for k in range(self.K):
            xs_r.append((1 + self.eps) * xs_r[-1] + self.propagate(edge_index, x=xs_r[-1]))
        xs_t = [x]
        for k in range(self.K):
            xs_t.append((1 + self.eps) * xs_t[-1] + self.propagate(edge_index_t, x=xs_t[-1]))

        xs_r = torch.cat(xs_r, dim=1)
        xs_t = torch.cat(xs_t, dim=1)
        out = torch.cat([xs_r, xs_t], dim=1)

        # out = self.nn((1 + self.eps) * x + self.propagate(edge_index, x=x))
        return out

    def message(self, x_j):
        return x_j

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)