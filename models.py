import torch as th
import torch.nn.functional as F
import dgl.nn as dglnn
from dgl.utils import expand_as_pair
from dgl import function as dglfn
from dgl.ops import edge_softmax

"""
The following code is taken directly from
https://github.com/mlfoundations/model-soups/blob/main/learned_bylayer.py
This is the code from "Model soups: averaging weights of multiple 
fine-tuned models improves accuracy without increasing inference time",
which is the paper that introduced learned souping.
"""
def del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])

def set_attr(obj, names, val):
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)

def make_functional(mod):
    orig_params = tuple(mod.parameters())
    # Remove all the parameters in the model
    names = []
    for name, p in list(mod.named_parameters()):
        del_attr(mod, name.split("."))
        names.append(name)
    return orig_params, names

def load_weights(mod, names, params):
    for name, p in zip(names, params):
        set_attr(mod, name.split("."), p)

class AlphaWrapper(th.nn.Module):
    def __init__(self, paramslist, model, names, device=th.device('cpu'), init='ones'):
        super(AlphaWrapper, self).__init__()
        self.device = device
        self.paramslist = paramslist
        self.model = model
        self.names = names
        if init == 'ones':
            ralpha = th.ones(len(paramslist[0]), len(paramslist))
        elif init == 'uniform':
            ralpha = th.nn.init.xavier_uniform_(th.ones(len(paramslist[0]), len(paramslist)))
        elif init == 'normal':
            ralpha = th.nn.init.xavier_normal_(th.ones(len(paramslist[0]), len(paramslist)))
        else:
            raise ValueError("Invalid init method")
        ralpha = th.nn.functional.softmax(ralpha, dim=1)
        self.alpha_raw = th.nn.Parameter(ralpha)
        self.beta = th.nn.Parameter(th.tensor(1.))

    def alpha(self):
        return th.nn.functional.softmax(self.alpha_raw, dim=1)
    
    def get_state_dict(self, device=th.device('cpu')):
        """
        Returns the state dict of the model, NOT the alpha wrapper.
        The returned state dict will be on the requested device.
        """
        alph = self.alpha()
        params = tuple(sum(tuple(pi * alphai for pi, alphai in zip(p, alph[j].cpu()))) for j, p in enumerate(zip(*self.paramslist)))
        # needs to be detached otherwise pytorch will call you a coward if you try to move a leaf tensor between processes
        params = tuple(p.to(device).detach() for p in params)
        state_dict = dict(zip(self.names, params))
        return state_dict
    
    def enable_minibatching(self):
        self.model.enable_minibatching()

    def forward(self, g, feats):
        alph = self.alpha()
        params = tuple(sum(tuple(pi * alphai for pi, alphai in zip(p, alph[j].cpu()))) for j, p in enumerate(zip(*self.paramslist)))
        params = tuple(p.to(self.device) for p in params)
        load_weights(self.model, self.names, params)
        out = self.model(g, feats)
        return self.beta * out


"""
The following code is taken directly from
https://github.com/dmlc/dgl/blob/master/examples/pytorch/ogb/ogbn-products/gat/models.py
This is the official DGL implementation of GAT.

We have made some slight modifications to it so that it fits into our codebase.
"""
class GATConv(th.nn.Module):
    def __init__(
        self,
        node_feats,
        edge_feats,
        out_feats,
        n_heads=1,
        attn_drop=0.0,
        edge_drop=0.0,
        negative_slope=0.2,
        residual=True,
        activation=None,
        use_attn_dst=True,
        allow_zero_in_degree=True,
        use_symmetric_norm=False,
    ):
        super(GATConv, self).__init__()
        self._n_heads = n_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(node_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._use_symmetric_norm = use_symmetric_norm

        # feat fc
        self.src_fc = th.nn.Linear(
            self._in_src_feats, out_feats * n_heads, bias=False
        )
        if residual:
            self.dst_fc = th.nn.Linear(self._in_src_feats, out_feats * n_heads)
            self.bias = None
        else:
            self.dst_fc = None
            self.bias = th.nn.Parameter(out_feats * n_heads)

        # attn fc
        self.attn_src_fc = th.nn.Linear(self._in_src_feats, n_heads, bias=False)
        if use_attn_dst:
            self.attn_dst_fc = th.nn.Linear(
                self._in_src_feats, n_heads, bias=False
            )
        else:
            self.attn_dst_fc = None
        if edge_feats > 0:
            self.attn_edge_fc = th.nn.Linear(edge_feats, n_heads, bias=False)
        else:
            self.attn_edge_fc = None

        self.attn_drop = th.nn.Dropout(attn_drop)
        self.edge_drop = edge_drop
        self.leaky_relu = th.nn.LeakyReLU(negative_slope, inplace=True)
        self.activation = activation

        self.reset_parameters()

    def reset_parameters(self):
        gain = th.nn.init.calculate_gain("relu")
        th.nn.init.xavier_normal_(self.src_fc.weight, gain=gain)
        if self.dst_fc is not None:
            th.nn.init.xavier_normal_(self.dst_fc.weight, gain=gain)

        th.nn.init.xavier_normal_(self.attn_src_fc.weight, gain=gain)
        if self.attn_dst_fc is not None:
            th.nn.init.xavier_normal_(self.attn_dst_fc.weight, gain=gain)
        if self.attn_edge_fc is not None:
            th.nn.init.xavier_normal_(self.attn_edge_fc.weight, gain=gain)

        if self.bias is not None:
            th.nn.init.zeros_(self.bias)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat_src, feat_edge=None):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    assert False

            if graph.is_block:
                feat_dst = feat_src[: graph.number_of_dst_nodes()]
            else:
                feat_dst = feat_src

            if self._use_symmetric_norm:
                degs = graph.out_degrees().float().clamp(min=1)
                norm = th.pow(degs, -0.5)
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = th.reshape(norm, shp)
                feat_src = feat_src * norm

            feat_src_fc = self.src_fc(feat_src).view(
                -1, self._n_heads, self._out_feats
            )
            feat_dst_fc = self.dst_fc(feat_dst).view(
                -1, self._n_heads, self._out_feats
            )
            attn_src = self.attn_src_fc(feat_src).view(-1, self._n_heads, 1)

            # NOTE: GAT paper uses "first concatenation then linear projection"
            # to compute attention scores, while ours is "first projection then
            # addition", the two approaches are mathematically equivalent:
            # We decompose the weight vector a mentioned in the paper into
            # [a_l || a_r], then
            # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            # Our implementation is much efficient because we do not need to
            # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
            # addition could be optimized with DGL's built-in function u_add_v,
            # which further speeds up computation and saves memory footprint.
            graph.srcdata.update(
                {"feat_src_fc": feat_src_fc, "attn_src": attn_src}
            )

            if self.attn_dst_fc is not None:
                attn_dst = self.attn_dst_fc(feat_dst).view(-1, self._n_heads, 1)
                graph.dstdata.update({"attn_dst": attn_dst})
                graph.apply_edges(
                    dglfn.u_add_v("attn_src", "attn_dst", "attn_node")
                )
            else:
                graph.apply_edges(dglfn.copy_u("attn_src", "attn_node"))

            e = graph.edata["attn_node"]
            if feat_edge is not None:
                attn_edge = self.attn_edge_fc(feat_edge).view(
                    -1, self._n_heads, 1
                )
                graph.edata.update({"attn_edge": attn_edge})
                e += graph.edata["attn_edge"]
            e = self.leaky_relu(e)

            if self.training and self.edge_drop > 0:
                perm = th.randperm(graph.num_edges(), device=e.device)
                bound = int(graph.num_edges() * self.edge_drop)
                eids = perm[bound:]
                graph.edata["a"] = th.zeros_like(e)
                graph.edata["a"][eids] = self.attn_drop(
                    edge_softmax(graph, e[eids], eids=eids)
                )
            else:
                graph.edata["a"] = self.attn_drop(edge_softmax(graph, e))

            # message passing
            graph.update_all(
                dglfn.u_mul_e("feat_src_fc", "a", "m"), dglfn.sum("m", "feat_src_fc")
            )
            rst = graph.dstdata["feat_src_fc"]

            if self._use_symmetric_norm:
                degs = graph.in_degrees().float().clamp(min=1)
                norm = th.pow(degs, 0.5)
                shp = norm.shape + (1,) * (feat_dst.dim())
                norm = th.reshape(norm, shp)
                rst = rst * norm

            # residual
            if self.dst_fc is not None:
                rst += feat_dst_fc
            else:
                rst += self.bias

            # activation
            if self.activation is not None:
                rst = self.activation(rst, inplace=True)

            return rst

class GAT(th.nn.Module):
    def __init__(
        self,
        node_feats,
        edge_feats,
        n_classes,
        n_layers,
        n_heads,
        n_hidden,
        edge_emb,
        activation,
        dropout,
        input_drop,
        attn_drop,
        edge_drop,
        use_attn_dst=True,
        allow_zero_in_degree=False,
        residual=False,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.num_layers = n_layers

        self.convs = th.nn.ModuleList()
        self.norms = th.nn.ModuleList()

        self.node_encoder = th.nn.Linear(node_feats, n_hidden)
        if edge_emb > 0:
            self.edge_encoder = th.nn.ModuleList()
        else:
            self.edge_encoder = None

        for i in range(n_layers):
            in_hidden = n_heads * n_hidden if i > 0 else node_feats
            out_hidden = n_hidden

            if self.edge_encoder is not None:
                self.edge_encoder.append(th.nn.Linear(edge_feats, edge_emb))
            self.convs.append(
                GATConv(
                    in_hidden,
                    edge_emb,
                    out_hidden,
                    n_heads=n_heads,
                    attn_drop=attn_drop,
                    edge_drop=edge_drop,
                    use_attn_dst=use_attn_dst,
                    allow_zero_in_degree=allow_zero_in_degree,
                )
            )
            self.norms.append(th.nn.BatchNorm1d(n_heads * out_hidden, track_running_stats=False))

        self.pred_linear = th.nn.Linear(n_heads * n_hidden, n_classes)

        self.input_drop = th.nn.Dropout(input_drop)
        self.dropout = th.nn.Dropout(dropout)
        self.activation = activation
        self.residual = residual

    def enable_minibatching(self):
        pass # This doesn't need to be implemented because I believe GAT supports both minibatching and non-minibatching without any manual settings

    def disable_minibatching(self):
        pass # This doesn't need to be implemented because I believe GAT supports both minibatching and non-minibatching without any manual settings

    def forward(self, g, h, inference=False):
        if not isinstance(g, list):
            subgraphs = [g] * self.n_layers
        else:
            subgraphs = g

        h = self.input_drop(h)

        h_last = None

        for i in range(self.n_layers):
            if self.edge_encoder is not None:
                efeat = subgraphs[i].edata["feat"]
                efeat_emb = self.edge_encoder[i](efeat)
                efeat_emb = F.relu(efeat_emb, inplace=True)
            else:
                efeat_emb = None

            h = self.convs[i](subgraphs[i], h, efeat_emb).flatten(1, -1)

            if self.residual and h_last is not None:
                h += h_last[: h.shape[0], :]

            h_last = h

            h = self.norms[i](h)
            h = self.activation(h, inplace=True)
            h = self.dropout(h)

            if inference:
                th.cuda.empty_cache()

        h = self.pred_linear(h)

        return h
    

"""
The following code is originally from https://github.com/dmlc/dgl/blob/master/examples/pytorch/gcn/train.py
It is the official DGL implementation of GCN. 
We have modified it heavily to support minibatching and other features.
"""
class GCN(th.nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layers, dropout, minibatching=True, edge_weight_key=None):
        """
        in_size: int - the number of input features
        hid_size: int - the number of hidden features
        out_size: int - the number of output features
        num_layers: int - the number of layers
        dropout: float - the dropout rate
        minibatching: bool - whether or not to use minibatching
        edge_weight_key: str - the key in the edge data dictionary that contains the edge weights. 
            If None, then then edge weights are not used (which has the same effect as setting all edge weights to 1.0)
        """
        super().__init__()
        self.layers = th.nn.ModuleList()
        self.dropout = th.nn.Dropout(dropout)
        self.minibatching_enabled = minibatching
        self.num_layers = num_layers
        self.edge_weight_key = edge_weight_key

        if num_layers == 1:
            self.layers.append(dglnn.GraphConv(in_size, out_size, activation=F.relu, allow_zero_in_degree=True))
        elif num_layers == 2:
            self.layers.append(dglnn.GraphConv(in_size, hid_size, activation=F.relu, allow_zero_in_degree=True))
            self.layers.append(dglnn.GraphConv(hid_size, out_size, allow_zero_in_degree=True))
        elif num_layers > 2:
            self.layers.append(dglnn.GraphConv(in_size, hid_size, activation=F.relu, allow_zero_in_degree=True))
            for i in range(num_layers - 2):
                self.layers.append(dglnn.GraphConv(hid_size, hid_size, activation=F.relu, allow_zero_in_degree=True))
            self.layers.append(dglnn.GraphConv(hid_size, out_size, allow_zero_in_degree=True))
        else:
            raise ValueError("Invalid number of layers.")
        
    def enable_minibatching(self):
        self.minibatching_enabled = True

    def disable_minibatching(self):
        self.minibatching_enabled = False

    def forward(self, graph, features):
        if self.edge_weight_key is not None:
            if self.minibatching_enabled:
                h = features
                for l, (layer, block) in enumerate(zip(self.layers, graph)):
                    if l != 0:
                        h = self.dropout(h)
                    h = layer(block, h, edge_weight=block.edata[self.edge_weight_key])
                return h
            else:
                edge_wts = graph.edata[self.edge_weight_key].to(graph.device) # for some reason, the edge weights need to be moved to the device of the graph even though they come from the graph
                h = self.layers[0](graph, features, edge_weight=edge_wts)
                for layer in self.layers[1:]:
                    h = self.dropout(h)
                    h = layer(graph, h, edge_weight=edge_wts)
                return h
        else:
            if self.minibatching_enabled:
                h = features
                for l, (layer, block) in enumerate(zip(self.layers, graph)):
                    if l != 0:
                        h = self.dropout(h)
                    h = layer(block, h)
                return h
            else:
                h = self.layers[0](graph, features)
                for layer in self.layers[1:]:
                    h = self.dropout(h)
                    h = layer(graph, h)
                return h


"""
The following is a custom implementation of GraphSAGE that supports minibatching.
"""
class MinibatchGraphSAGE(th.nn.Module):
    """
    A version of GraphSAGE that supports minibatching. Minibatching can be enabled or disabled
    using the methods enable_minibatching() and disable_minibatching(). When minibatching is
    enabled, the input graph must be a list of DGLGraphs, and the input features must be a
    list of tensors, one for each graph in the minibatch. When minibatching is disabled, the
    input graph must be a single DGLGraph, and the input features must be a single tensor.
    """
    def __init__(self, in_feats, hidden_feats, dropout, aggregator, num_classes, num_layers, minibatching=True):
        super(MinibatchGraphSAGE, self).__init__()
        self.layers = th.nn.ModuleList()

        self.hid_size = hidden_feats
        self.out_size = num_classes
        self.num_layers = num_layers

        if num_layers == 1:
            self.layers.append(dglnn.SAGEConv(in_feats=in_feats, 
                                              out_feats=num_classes, 
                                              aggregator_type=aggregator))
        elif num_layers == 2:
            self.layers.append(dglnn.SAGEConv(in_feats=in_feats, 
                                              out_feats=hidden_feats, 
                                              aggregator_type=aggregator,
                                              activation=F.relu))
            self.layers.append(dglnn.SAGEConv(in_feats=hidden_feats, 
                                              out_feats=num_classes, 
                                              aggregator_type=aggregator))
        elif num_layers > 2:
            self.layers.append(dglnn.SAGEConv(in_feats=in_feats, 
                                            out_feats=hidden_feats, 
                                            aggregator_type=aggregator,
                                            feat_drop=dropout,
                                            activation=F.relu))
            for i in range(num_layers - 2):
                self.layers.append(dglnn.SAGEConv(in_feats=hidden_feats, 
                                                out_feats=hidden_feats, 
                                                aggregator_type=aggregator,
                                                feat_drop=dropout,
                                                activation=F.relu))
            self.layers.append(dglnn.SAGEConv(in_feats=hidden_feats, 
                                            out_feats=num_classes, 
                                            aggregator_type=aggregator))
        else:
            raise ValueError("Invalid number of layers.")

        self.minibatching_enabled = minibatching

    def enable_minibatching(self):
        self.minibatching_enabled = True

    def disable_minibatching(self):
        self.minibatching_enabled = False

    def forward(self, graph, features):
        if self.minibatching_enabled:
            h = features
            for l, (layer, block) in enumerate(zip(self.layers, graph)):
                h = layer(block, h)
            return h
        else:
            logits = self.layers[0](graph, features)
            for layer in self.layers[1:]:
                logits = layer(graph, logits)
            return logits