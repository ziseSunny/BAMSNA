import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn.conv import MessagePassing, GCNConv
from torch_geometric.utils import add_remaining_self_loops, degree

class DirectedGCN(MessagePassing):
    def __init__(self, cached=False, bias=True, **kwargs):
        super(DirectedGCN, self).__init__(aggr='add', **kwargs)

    def forward(self, x, edge_index, edge_weight=None):

        # 定义归一化处理的方式
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=x.dtype,
                                     device=edge_index.device)
        fill_value = 1
        edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, fill_value, num_nodes=x.size(0))

        row, col = edge_index
        deg_r = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt_r = deg_r.pow(-1.0)
        deg_inv_sqrt_r[deg_inv_sqrt_r == float('inf')] = 0

        edge_index_t = torch.vstack([edge_index[1,:],edge_index[0,:]])
        row_t, col_t = edge_index_t

        deg_t = degree(row_t, x.size(0), dtype=x.dtype)
        deg_inv_sqrt_t = deg_t.pow(-1.0)
        deg_inv_sqrt_t[deg_inv_sqrt_t == float('inf')] = 0

        norm_r = deg_inv_sqrt_r[row] * edge_weight
        norm_t = deg_inv_sqrt_t[row_t] * edge_weight

        # edge_index, norm = GCNConv.norm(edge_index, x.size(0), edge_weight,dtype=x.dtype)

        xs_r = self.propagate(edge_index, x=x[:,:int(x.size(1)/2)], norm=norm_r)
        xs_t = self.propagate(edge_index, x=x[:,int(x.size(1)/2):], norm=norm_t)
        xs = torch.cat([xs_r, xs_t], dim=1)
        return xs

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, {}, K={})'.format(self.__class__.__name__,
                                         self.in_channels, self.out_channels,
                                         self.K)