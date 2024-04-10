import torch as th
import dgl
from ogb.nodeproppred import DglNodePropPredDataset
from tqdm import tqdm
from models import MinibatchGraphSAGE, GCN, GAT
import torch.nn.functional as F

"""
Graph/GNN helper functions
"""
def get_train_mask(graph):
    """
    Returns the train mask of the graph
    """
    if "_N/train_mask" in graph.ndata.keys():
        train_mask = graph.ndata["_N/train_mask"]
        train_mask = train_mask.type(th.bool)
    elif "train_mask" in graph.ndata.keys():
        train_mask = graph.ndata["train_mask"]
        train_mask = train_mask.type(th.bool)
    else:
        print("The graph training mask could not be found.")
        print("The graph keys are:")
        print(graph.ndata.keys())
        raise KeyError
    return train_mask

def get_val_mask(graph):
    """
    Returns the validation mask of the graph
    """
    if "_N/val_mask" in graph.ndata.keys():
        val_mask = graph.ndata["_N/val_mask"]
        val_mask = val_mask.type(th.bool)
    elif "val_mask" in graph.ndata.keys():
        val_mask = graph.ndata["val_mask"]
        val_mask = val_mask.type(th.bool)
    else:
        print("The graph validation mask could not be found.")
        print("The graph keys are:")
        print(graph.ndata.keys())
        raise KeyError
    return val_mask

def get_test_mask(graph):
    """
    Returns the test mask of the graph
    """
    if "_N/test_mask" in graph.ndata.keys():
        test_mask = graph.ndata["_N/test_mask"]
        test_mask = test_mask.type(th.bool)
    elif "test_mask" in graph.ndata.keys():
        test_mask = graph.ndata["test_mask"]
        test_mask = test_mask.type(th.bool)
    else:
        print("The graph test mask could not be found.")
        print("The graph keys are:")
        print(graph.ndata.keys())
        raise KeyError
    return test_mask

def get_features(graph):
    """
    Returns the features of the graph
    """
    if "_N/feat" in graph.ndata.keys():
        features = graph.ndata["_N/feat"]
    elif "feat" in graph.ndata.keys():
        features = graph.ndata["feat"]
    else:
        print("The graph features could not be found.")
        print("The graph keys are:")
        print(graph.ndata.keys())
        raise KeyError
    return features

def get_labels(graph):
    """
    Returns the labels of the graph
    """
    if "_N/labels" in graph.ndata.keys():
        labels = graph.ndata["_N/labels"]
    elif "labels" in graph.ndata.keys():
        labels = graph.ndata["labels"]
    elif "label" in graph.ndata.keys():
        labels = graph.ndata["label"]
    else:
        print("The graph labels could not be found.")
        print("The graph keys are:")
        print(graph.ndata.keys())
        raise KeyError
    return labels

def get_num_classes(graph):
    """
    Returns the number of classes in the graph
    """
    labels = get_labels(graph)
    return len(th.unique(labels[th.logical_not(th.isnan(labels))]))

def get_in_feats(graph):
    """
    Returns the number of input features of the graph
    """
    features = get_features(graph)
    return features.shape[1]

def get_num_edge_feats(graph, force_nonzero=False, execute_silently=False):
    """
    Returns the number of edge features of the graph
    If force_nonzero is true, then if the graph has no edge features, 
    an error will be raised.
    If force_nonzero is false, then if the graph has no edge features,
    0 will be returned.
    If execute_silently is true, then no print statements will be executed.
    """
    if "_E/feat" in graph.edata.keys():
        return graph.edata["_E/feat"].shape[1]
    elif "feat" in graph.edata.keys():
        return graph.edata["feat"].shape[1]
    else:
        if not execute_silently:
            print("The graph edge features could not be found.")
        if force_nonzero:
            print("The graph.edata keys are:")
            print(graph.edata.keys())
            raise KeyError
        else:
            if not execute_silently:
                print("Using 0 edge features.")
            return 0

def get_edge_embedding_size(graph, force_nonzero=False, execute_silently=False):
    """
    Returns the size of the edge embedding of the graph.
    If the graph has no edge features, returns 0,
    unless force_nonzero is true, in which case an error is raised.
    """
    if get_num_edge_feats(graph, force_nonzero, execute_silently=execute_silently) == 0:
        return 0
    else:
        print("The graph has edge features, but getting edge embeddings is not yet supported.")
        raise NotImplementedError

def get_train_nids(graph):
    """
    Returns the node ids of the graph from the mask
    """
    if "_N/train_mask" in graph.ndata.keys():
        nids = th.where(graph.ndata["_N/train_mask"] == True)[0]
    elif "train_mask" in graph.ndata.keys():
        nids = th.where(graph.ndata["train_mask"] == True)[0]
    else:
        print("The graph training mask could not be found.")
        print("The graph keys are:")
        print(graph.ndata.keys())
        raise KeyError
    return nids

def get_val_nids(graph):
    """
    Returns the node ids of the graph from the mask
    """
    if "_N/val_mask" in graph.ndata.keys():
        nids = th.where(graph.ndata["_N/val_mask"] == True)[0]
    elif "val_mask" in graph.ndata.keys():
        nids = th.where(graph.ndata["val_mask"] == True)[0]
    else:
        print("The graph validation mask could not be found.")
        print("The graph keys are:")
        print(graph.ndata.keys())
        raise KeyError
    return nids

def get_test_nids(graph):
    """
    Returns the node ids of the graph from the mask
    """
    if "_N/test_mask" in graph.ndata.keys():
        nids = th.where(graph.ndata["_N/test_mask"] == True)[0]
    elif "test_mask" in graph.ndata.keys():
        nids = th.where(graph.ndata["test_mask"] == True)[0]
    else:
        print("The graph test mask could not be found.")
        print("The graph keys are:")
        print(graph.ndata.keys())
        raise KeyError
    return nids

def get_node_feature_key(graph):
    if 'feat' in graph.ndata.keys():
        feature_key = 'feat'
    elif 'feats' in graph.ndata.keys():
        feature_key = 'feats'
    elif '_N/feat' in graph.ndata.keys():
        feature_key = '_N/feat'
    elif '_N/feats' in graph.ndata.keys():
        feature_key = '_N/feats'
    else:
        raise ValueError("The graph does not have a feature key")
    
    return feature_key

def get_node_label_key(graph):
    if 'label' in graph.ndata.keys():
        label_key = 'label'
    elif 'labels' in graph.ndata.keys():
        label_key = 'labels'
    elif '_N/label' in graph.ndata.keys():
        label_key = '_N/label'
    elif '_N/labels' in graph.ndata.keys():
        label_key = '_N/labels'
    else:
        raise ValueError("The graph does not have a label key")
    
    return label_key

def split_boolean_mask(boolean_mask, device=th.device('cpu')):
    """
    boolean_mask is a tensor of booleans
    this function returns two tensors of the same dimension
    the first tensor contains half of the true values of val_mask
    the second tensor contains the other half of the true values of val_mask

    Currently this function is only used to split the validation set in half,
    which is how we perform cross validation when souping.
    """

    # get a random boolean tensor of the same shape as val_mask 
    # (half of the values will be true), half will be false
    random_mask = th.rand(boolean_mask.shape, device=device) > 0.5
    inverse_random_mask = th.logical_not(random_mask)

    # get the two tensors
    first_tensor = th.logical_and(boolean_mask, random_mask)
    second_tensor = th.logical_and(boolean_mask, inverse_random_mask)

    return first_tensor, second_tensor

def load_graph_dataset(graph_name):
    """
    Loads the unchanged, entire base graph for the given graph name.
    """
    if graph_name == 'reddit':
        data = dgl.data.RedditDataset()
        graph = data[0]
    elif graph_name == 'flickr':
        data = dgl.data.FlickrDataset()
        graph = data[0]
    elif graph_name == "ogbn-papers100M":
        if True:
            """
            Load the preprocessed graph (already made bidirectional, etc.)
            """
            graph = dgl.load_graphs("./papers_128_parts/ogbn-papers100M_dgl_graph.bin")[0][0]
        else: #old code, kept for reference
            print(f"Loading {graph_name} graph")
            data = DglNodePropPredDataset(name=graph_name)
            graph, labels = data[0]
            
            print("Making graph bidirectional by adding reverse edges")
            u, v = graph.all_edges() # u and v are the source and destination nodes of the edges
            inverse_mask = graph.has_edges_between(v, u) # mask is True for bidirectional edges
            mask = ~inverse_mask # mask is True for unidirectional edges
            non_bi_u = u[mask] # non_bi_u now contains only the unidirectional edges
            non_bi_v = v[mask] # non_bi_v now contains only the unidirectional edges
            graph.add_edges(non_bi_v, non_bi_u) # add the reverse edges to make the graph bidirectional
            """
            Short note on this:
            1. OGBN-Papers100M tends to get preprocessed to have undirected edges,
            which results in higher scores.
            2. Typically making a graph undirected is done by using DGL's 
            to_bidirected function. However, this function seems to hang when used
            with OGBN-Papers100M. This is likely due to the large size of the graph.
            3. OGBN-Papers100M contains no self-loops and no edge features, so adding
            reverse edges by just adding non-bidirectional edges in reverse order is 
            safe. DGLGraphs.add_edges() adds duplicate edges, and graph.to_simple()
            is also far too slow/memory intense, so this code finds which edges are not
            bidirectional (3247032 ARE bidirectional, the rest are now), and then adds 
            the reverse edges. This is a bit of a hack, but it works.
            """

            labels = labels.view(-1).type(th.long)
            splitted_idx = data.get_idx_split()
            train_idx, val_idx, test_idx = splitted_idx["train"], splitted_idx["valid"], splitted_idx["test"]
            train_mask = th.zeros((graph.num_nodes(),), dtype=th.bool)
            train_mask[train_idx] = True
            val_mask = th.zeros((graph.num_nodes(),), dtype=th.bool)
            val_mask[val_idx] = True
            test_mask = th.zeros((graph.num_nodes(),), dtype=th.bool)
            test_mask[test_idx] = True
            graph.ndata['labels'] = labels
            graph.ndata['train_mask'] = train_mask
            graph.ndata['val_mask'] = val_mask
            graph.ndata['test_mask'] = test_mask
    else:
        data = DglNodePropPredDataset(name=graph_name)
        graph, labels = data[0]
        labels = labels[:, 0]
        graph.ndata['labels'] = labels
        
        splitted_idx = data.get_idx_split()
        train_nid, val_nid, test_nid = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']
        train_mask = th.zeros((graph.num_nodes(),), dtype=th.bool)
        train_mask[train_nid] = True
        val_mask = th.zeros((graph.num_nodes(),), dtype=th.bool)
        val_mask[val_nid] = True
        test_mask = th.zeros((graph.num_nodes(),), dtype=th.bool)
        test_mask[test_nid] = True
        graph.ndata['train_mask'] = train_mask
        graph.ndata['val_mask'] = val_mask
        graph.ndata['test_mask'] = test_mask

    # OGBN-ArXiv is usually pre-processed to have undirected edges
    if graph_name == 'ogbn-arxiv':
        graph = dgl.to_bidirected(graph, copy_ndata=True)

    return graph

def create_proper_model(graph, model_to_use, num_layers, hidden_layer_size, dropout, aggregator_type=None, num_heads=None, input_drop=None, attn_drop=None, edge_drop=None, use_attn_dst=None, execute_silently=False):
    """
    Creates a model based on the given parameters.

    Currently supports GraphSAGE, GCN, and GAT
    __Parameters__
    graph : the graph to train on
    model_to_use : the model to use. Options are graphsage, gcn, and gat
    num_layers : the number of total layers in the model
    hidden_layer_size : the size of the hidden layers
    dropout : the dropout rate
    aggregator_type : the type of aggregator to use. Only used for GraphSAGE
    num_heads : the number of attention heads. Only used for GAT
    input_drop : the input dropout rate. Only used for GAT
    attn_drop : the attention dropout rate. Only used for GAT
    edge_drop : the edge dropout rate. Only used for GAT
    use_attn_dst : whether to use attention on the destination nodes. Only used for GAT
    """

    in_feats = get_in_feats(graph)
    out_feats = get_num_classes(graph)
    num_edge_feats = get_num_edge_feats(graph, execute_silently=execute_silently)
    edge_embedding = get_edge_embedding_size(graph, execute_silently=execute_silently)

    if model_to_use == 'graphsage':
        if aggregator_type is None:
            print("The model " + model_to_use + " requires aggregator_type to be specified.")
            raise ValueError

        model = MinibatchGraphSAGE(in_feats=in_feats,
                                    hidden_feats=hidden_layer_size,
                                    dropout=dropout,
                                    aggregator=aggregator_type,
                                    num_classes=out_feats,
                                    num_layers=num_layers,
                                    minibatching=True)
    elif model_to_use == 'gcn':
        model = GCN(in_size=in_feats,
                    hid_size=hidden_layer_size,
                    out_size=out_feats,
                    num_layers=num_layers,
                    dropout=dropout,
                    minibatching=True)
    elif model_to_use == 'gat':
        if num_heads is None:
            print("The model " + model_to_use + " requires num_heads to be specified.")
            raise ValueError
        if input_drop is None:
            print("The model " + model_to_use + " requires input_drop to be specified.")
            raise ValueError
        if attn_drop is None:
            print("The model " + model_to_use + " requires attn_drop to be specified.")
            raise ValueError
        if edge_drop is None:
            print("The model " + model_to_use + " requires edge_drop to be specified.")
            raise ValueError
        if use_attn_dst is None:
            print("The model " + model_to_use + " requires use_attn_dst to be specified.")
            raise ValueError

        model = GAT(node_feats=in_feats,
                    edge_feats=num_edge_feats,
                    n_classes=out_feats,
                    n_layers=num_layers,
                    n_heads=num_heads,
                    n_hidden=hidden_layer_size,
                    edge_emb=edge_embedding,
                    activation=F.relu,
                    dropout=dropout,
                    input_drop=input_drop,
                    attn_drop=attn_drop,
                    edge_drop=edge_drop,
                    use_attn_dst=use_attn_dst,
                    allow_zero_in_degree=True,
                    residual=False
                    )
    else:
        print("The model " + model_to_use + " is not supported.")
        raise ValueError
        
    return model

"""
Evaluation functions
"""
@staticmethod
def count_corrects(pred: th.Tensor, label: th.Tensor) -> int:
    assert pred.dim() == 1 and label.dim() == 1 and pred.shape == label.shape
    return ((pred == label) + 0.0).sum().item()

def test(g, model, device=th.device('cpu')):
    """
    Evaluate the model on the full test set. Does not produce gradients.
    Parameters:
        g : the entire graph (not partitioned)
        model : the model to evaluate
        device : the device to evaluate on
    returns the accuracy on the entire test set
    """

    # move the model and graph to the device
    model.to(device)
    g = g.to(device)

    # get the information needed for evaluation on the test set
    features = get_features(g)
    labels = get_labels(g)
    test_mask = get_test_mask(g)

    # prepare the model
    model.eval()
    model.disable_minibatching()

    with th.no_grad():
        # Forward
        logits = model(g, features)

        # Compute prediction
        pred = logits.argmax(1)

        # Compute accuracy on test set
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()
        return test_acc

def batched_test(g, model, neighbors_to_sample_per_layer, batch_size, workers_for_data_loading=0, device=th.device('cpu'), print_progress=True):
    """
    Evaluate the model on the full test set, performing neighbor sampling
    to reduce memory footprint. Does not produce gradients.
    Parameters:
        g : the entire graph (not partitioned)
        model : the model to evaluate
        neighbors_to_sample_per_layer : the number of neighbors to sample per layer
        device : the device to evaluate on
    returns the accuracy on the entire test set
    """

    # move the model and graph to the device
    model.to(device)

    # get the information needed for evaluation on the validation set
    test_mask = get_test_mask(g)
    labels = get_labels(g)
    test_nids = get_test_nids(g)

    # prepare the model
    model.eval()
    model.enable_minibatching()

    feature_key = get_node_feature_key(g)
    label_key = get_node_label_key(g)

    # create the sampler
    if neighbors_to_sample_per_layer == -1:
        neighbor_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(model.num_layers)
    else:
        if type(neighbors_to_sample_per_layer) == int:
            fanout = [neighbors_to_sample_per_layer for _ in range(model.num_layers)]
        elif type(neighbors_to_sample_per_layer) == list and len(neighbors_to_sample_per_layer) == model.num_layers:
            fanout = neighbors_to_sample_per_layer
        else:
            raise ValueError("neighbors_to_sample_per_layer must be an int or a list of length model.num_layers")
        neighbor_sampler = dgl.dataloading.NeighborSampler(fanout,
                                                        prefetch_node_feats=[feature_key],
                                                        prefetch_labels=[label_key])
        
    # create the dataloader
    test_dataloader = dgl.dataloading.DataLoader(
        # The following arguments are specific to DGL's DataLoader.
        graph=g,                                # The graph
        indices=test_nids,                      # The node IDs to iterate over in minibatches
        graph_sampler=neighbor_sampler,         # The neighbor sampler
        device=device,                          # Put the sampled MFGs on CPU or GPU
        # The following arguments are inherited from PyTorch DataLoader.
        batch_size=batch_size,                  # Batch size
        shuffle=False,                          # Whether to shuffle the nodes for every epoch
        drop_last=False,                        # Whether to drop the last incomplete batch
        num_workers=workers_for_data_loading    # Number of sampler processes
    )

    correct_cnt = 0
    with th.no_grad():
        if print_progress:
            iterator = tqdm(test_dataloader, desc='test')
        else:
            iterator = test_dataloader
            
        for input_nodes, output_nodes, blocks in iterator:
            blocks = [block.to(device) for block in blocks]
            inputs = blocks[0].srcdata[feature_key].to(device)
            labels = blocks[-1].dstdata[label_key].to(device)
            outputs = model(blocks, inputs)
            pred = th.topk(outputs, k=1).indices.view(-1)
            correct_cnt += count_corrects(pred, labels)
            
        num_test_nodes = th.sum(test_mask).item()
        test_acc = correct_cnt / num_test_nodes
        return test_acc

def validate(g, model, device=th.device('cpu')):
    """
    Evaluate the model on the full validation set. Does not produce gradients.
    Parameters:
        g : the entire graph (not partitioned)
        model : the model to evaluate
        device : the device to evaluate on
    returns the accuracy on the entire validation set
    """

    # move the model and graph to the device
    model.to(device)
    g = g.to(device)

    # get the information needed for evaluation on the validation set
    features = get_features(g)
    labels = get_labels(g)
    val_mask = get_val_mask(g)

    # prepare the model
    model.eval()
    model.disable_minibatching()

    with th.no_grad():
        # Forward
        logits = model(g, features)

        # Compute prediction
        pred = logits.argmax(1)

        # Compute accuracy on validation set
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        return val_acc