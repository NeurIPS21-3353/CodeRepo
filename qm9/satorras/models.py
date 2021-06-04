import torch
from torch import nn

from qm9.satorras.gcl import E_GCL, unsorted_segment_sum


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)

class BatchNorm1dMask(nn.Module):
    def __init__(self, nf):
        super(BatchNorm1dMask, self).__init__()
        self.bn = nn.BatchNorm1d(nf)

    def forward(self, x, mask):
        indexes = mask.squeeze(1).type(torch.bool).detach()
        masked_x = x[indexes]
        masked_x = self.bn(masked_x)
        x[indexes] = masked_x
        return x


class Node_MLP(nn.Module):
    def __init__(self, in_nf, hidden_nf, out_nf, bn=False, act_fn=nn.LeakyReLU(0.2)):
        super(Node_MLP, self).__init__()
        self.l1 = nn.Linear(in_nf, hidden_nf)
        self.bn = bn
        if self.bn:
            self.masked_bn = BatchNorm1dMask(hidden_nf)
        self.act_fn=act_fn
        self.l2 = nn.Linear(hidden_nf, out_nf)
    def forward(self, x, node_mask=None):
        x = self.l1(x)
        if self.bn:
            x = self.masked_bn(x, node_mask)
        x = self.act_fn(x)
        return self.l2(x)



'''
class E_GCL_mask(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, input_nf, output_nf, hidden_nf, K=None, edges_in_d=0, act_fn=nn.ReLU(), recurrent=True, coords_weight=1.0):
        super(E_GCL_mask, self).__init__()
        input_edge = input_nf * 2
        self.coords_weight = coords_weight
        self.recurrent = recurrent
        edge_coords_nf = 1


        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf),
            act_fn)

        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 1))

        if recurrent:
            self.gru = nn.GRUCell(hidden_nf, hidden_nf)


    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        return self.edge_mlp(out)

    def node_model(self, x, edge_index, edge_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        out = torch.cat([x, agg], dim=1)
        out = self.node_mlp(out)
        if self.recurrent:
            out = self.gru(agg, out)
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        coord += agg*self.coords_weight
        return coord

    def forward(self, h, edge_index, coord, node_mask, edge_mask, edge_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        edge_feat = edge_feat * edge_mask

        # TO DO: edge_feat = edge_feat * edge_mask

        #coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat)

        # TO DO: h = h * node_mask

        # coord = self.node_coord_model(h, coord)
        # x = self.node_model(x, edge_index, x[col], u, batch)  # GCN
        return h, coord, edge_attr

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.mean((coord_diff)**2, 1).unsqueeze(1)

        return radial, coord_diff

def unsorted_segment_sum(data, segment_ids, num_segments):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


'''


class E_GCL_mask(E_GCL):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, input_nf, output_nf, hidden_nf, K=None, edges_in_d=0, nodes_attr_dim=0, act_fn=nn.ReLU(), recurrent=True, coords_weight=1.0, attention=False, bn=False):
        E_GCL.__init__(self, input_nf, output_nf, hidden_nf, K=K, edges_in_d=edges_in_d, nodes_att_dim=nodes_attr_dim, act_fn=act_fn, recurrent=recurrent, coords_weight=coords_weight, attention=attention)

        del self.coord_mlp
        self.act_fn = act_fn
        '''
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_attr_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))
        '''
        input_edge = input_nf * 2
        edge_coords_nf = 1

        self.node_mlp = Node_MLP(hidden_nf + input_nf + nodes_attr_dim, hidden_nf, output_nf, bn=bn, act_fn=act_fn)

        '''
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 1),
            nn.Tanh())
        '''

        if attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid()
            )
            '''
            self.edge_mlp = nn.Sequential(
                nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf, hidden_nf))
            '''


    '''
    def edge_model(self, source, target, radial, edge_attr, edge_mask, n_nodes):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            softmax_mask = (1 - edge_mask) * 1e14
            att_a = self.att_mlp(out)
            att_a -= softmax_mask
            att_a = att_a.view(-1, n_nodes)
            att_alpha = F.softmax(att_a, 1)
            att_alpha = att_alpha.view(-1, 1)

            out = out * att_alpha
        return out
    '''

    def node_model(self, x, edge_index, edge_attr, node_attr, node_mask):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg, node_mask)
        # out = self.node_mlp(agg)
        if self.recurrent:
            out = x + out
            #out = self.gru(out, x)
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat, edge_mask):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat) * edge_mask
        agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        coord += agg*self.coords_weight
        return coord

    def forward(self, h, edge_index, coord, node_mask, edge_mask, edge_attr=None, node_attr=None, n_nodes=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)

        #edge_feat = self.edge_model(h[row], h[col], radial, edge_attr, edge_mask, n_nodes)
        edge_feat = edge_feat * edge_mask

        # TO DO: edge_feat = edge_feat * edge_mask

        #coord = self.coord_model(coord, edge_index, coord_diff, edge_feat, edge_mask)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr, node_mask)

        return h, coord, edge_attr




class EGNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=4, coords_weight=1.0, attention=False, bn=False, node_attr=1, dropout=0, output_dim=1):
        super(EGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.output_dim = output_dim
        #self.reg = reg
        ### Encoder
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        self.node_attr = node_attr
        if node_attr:
            n_node_attr = in_node_nf
        else:
            n_node_attr = 0
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL_mask(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, nodes_attr_dim=n_node_attr, act_fn=act_fn, recurrent=True, coords_weight=coords_weight, attention=attention, bn=bn))

        self.node_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                      act_fn,
                                      nn.Linear(self.hidden_nf, self.hidden_nf))
        # self.node_dec = Node_MLP(self.hidden_nf, self.hidden_nf, self.hidden_nf, bn=True, act_fn=act_fn)

        if dropout:
            self.graph_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                           nn.Dropout(0.2),
                                           nn.Linear(self.hidden_nf, self.output_dim))
        else:
            self.graph_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                           act_fn,
                                           nn.Linear(self.hidden_nf, self.output_dim))
        self.to(self.device)

    def forward(self, h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes):
        h = self.embedding(h0)
        for i in range(0, self.n_layers):
            if self.node_attr:
                h, _, _ = self._modules["gcl_%d" % i](h, edges, x, node_mask, edge_mask, edge_attr=edge_attr, node_attr=h0, n_nodes=n_nodes)
            else:
                h, _, _ = self._modules["gcl_%d" % i](h, edges, x, node_mask, edge_mask, edge_attr=edge_attr,
                                                      node_attr=None, n_nodes=n_nodes)
            #coords -= self.reg * coords


        h = self.node_dec(h)

        h = h * node_mask
        h = h.view(-1, n_nodes, self.hidden_nf)
        h = torch.sum(h, dim=1)
        pred = self.graph_dec(h)
        return pred.squeeze(1)



class GCL_schnet(E_GCL):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, input_nf, output_nf, hidden_nf, K=None, edges_in_d=0, nodes_attr_dim=0, act_fn=nn.ReLU(), recurrent=True, coords_weight=1.0, attention=False):
        E_GCL.__init__(self, input_nf, output_nf, hidden_nf, K=K, edges_in_d=edges_in_d, nodes_att_dim=nodes_attr_dim, act_fn=act_fn, recurrent=recurrent, coords_weight=coords_weight, attention=attention)

        # self.node_mlp = nn.Sequential(
        #     nn.Linear(hidden_nf + input_nf, hidden_nf),
        #     act_fn,
        #     nn.Linear(hidden_nf, hidden_nf),
        #     nn.BatchNorm1d(hidden_nf),
        #     act_fn,
        #     nn.Linear(hidden_nf, output_nf),
        #     nn.BatchNorm1d(output_nf),
        #     act_fn
        # )
        del self.coord_mlp
        self.linear_prev = nn.Linear(64, 64)
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

    def forward(self, h, edge_index, coord, node_mask, edge_mask, edge_attr=None, node_attr=None):

        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        h = self.linear_prev(h)
        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        edge_feat = edge_feat * edge_mask

        # TO DO: edge_feat = edge_feat * edge_mask

        #coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, edge_attr

    def node_model(self, h, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=h.size(0))
        if node_attr is not None:
            agg = torch.cat([h, agg], dim=1)
        else:
            agg = torch.cat([h, agg], dim=1)
        out = self.node_mlp(agg)
        if self.recurrent:
            out = h + out
            #out = self.gru(out, x)
        return out, agg

