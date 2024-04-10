import os
import torch as th
import dgl
import argparse
import time
from tqdm import tqdm
import tools
import torch.multiprocessing as mp

# Suppress the warning about TypedStorage being deprecated in DGL
# this is due to ml-gpu not being updated to the latest version of DGL
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated') #TODO: remove this before pushing to git

def train_full_set_minibatched(graph, model, epochs, lr, wd, train_nids, batch_size, sampled_neighbors, print_progress=True, device='cpu'):
    """
    Trains the given model on the full training set with minibatching.
    rank : the rank of the worker
    graph : the graph to train on
    model : the model to train
    epochs : the number of epochs to train for
    lr : the learning rate
    wd : the weight decay
    train_nids : the node IDs of the training nodes
    batch_size : the batch size
    sampled_neighbors : the number of neighbors sampled at each layer
    print_progress : whether to print the progress
    device : the device to train on
    """

    # move the model to the device. Prepare for training
    model.to(device)
    model.train()
    model.enable_minibatching()

    node_feat_key = tools.get_node_feature_key(graph)
    node_label_key = tools.get_node_label_key(graph)

    # create the sampler
    if sampled_neighbors == -1:
        neighbor_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(model.num_layers)
    else:
        fanout = [sampled_neighbors for i in range(model.num_layers)]
        neighbor_sampler = dgl.dataloading.MultiLayerNeighborSampler(fanout,
                                                                     prefetch_node_feats=[node_feat_key],
                                                                     prefetch_labels=[node_label_key])
    
    if batch_size == -1:
        batch_size = len(train_nids)

    # create the dataloader
    train_dataloader = dgl.dataloading.DataLoader(
        # The following arguments are specific to DGL's DataLoader.
        graph=graph,              # The graph
        indices=train_nids,         # The node IDs to iterate over in minibatches
        graph_sampler=neighbor_sampler,            # The neighbor sampler
        device=device,      # Put the sampled MFGs on CPU or GPU
        # The following arguments are inherited from PyTorch DataLoader.
        batch_size=batch_size,    # Batch size
        shuffle=True,       # Whether to shuffle the nodes for every epoch
        drop_last=False,    # Whether to drop the last incomplete batch
        num_workers=0       # Number of sampler processes
    )

    optimizer = th.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fcn = th.nn.CrossEntropyLoss()
    loss_fcn.to(device)

    for e in range(epochs):
        iterator = tqdm(enumerate(train_dataloader), desc=f"Epoch {e}", total=len(train_dataloader)) if print_progress else enumerate(train_dataloader)
        for step, (input_nodes, output_nodes, mfgs) in iterator:
            # feature copy from CPU to GPU takes place here
            mfgs = [x.to(device) for x in mfgs]

            inputs = mfgs[0].srcdata[node_feat_key].to(device)

            labels = mfgs[-1].dstdata[node_label_key].to(device)

            # Forward
            predictions = model(mfgs, inputs)

            # Compute loss
            loss = loss_fcn(predictions, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def ingredient_trainer_worker(rank, queue, initial_model, save_directory, graph, epochs, lr, wd, batch_size, sampled_neighbors, model_to_use, hidden_layer_size, dropout, num_layers, num_heads, input_drop, attn_drop, edge_drop, use_attn_dst, aggregator_type, test_batch_size, test_fanout, full_graph_on_device=False):
    """
    Trains the given model on the full training set with minibatching.
    rank : the rank of the worker
    queue : the queue of ingredient numbers to train
    initial_model : the initial model to start from
    save_directory : the directory to save the model to
    graph : the graph to train on
    epochs : the number of epochs to train for
    lr : the learning rate
    wd : the weight decay
    batch_size : the batch size
    sampled_neighbors : the number of sampled neighbors
    model_to_use : the model to use
    hidden_layer_size : the size of the hidden layers
    dropout : the dropout to use for all non-input layers
    num_layers : the number of layers to use
    num_heads : the number of heads to use (GAT)
    input_drop : the input dropout to use (GAT)
    attn_drop : the attention dropout to use (GAT)
    edge_drop : the edge dropout to use (GAT)
    use_attn_dst : whether or not to use attention on the destination nodes (GAT)
    aggregator_type : the aggregator type to use for all SageConv layers (GraphSAGE)
    test_batch_size : the batch size to use for testing (-1 for full-batch testing)
    test_fanout : the fanout to use for testing (-1 for full-neighbors)
    full_graph_on_device : whether or not the full graph is to be kept on the device.
     This decreases time to train at the cost of GPU memory usage.  (default is False)
    """

    # set the device
    device = th.device(f'cuda:{rank}')
    th.cuda.set_device(device) # set the device
    th.cuda.empty_cache() # clear the cache, prepare for training

    # set the train_nids
    train_nids = tools.get_train_nids(graph)

    # move the graph to the device if full_graph_on_device is true
    if full_graph_on_device:
        graph = graph.to(device)
        train_nids = train_nids.to(device)

    # set batch size if it is -1
    if batch_size == -1:
        batch_size = len(graph.nodes())

    print(f"[{rank}] Starting ingredient training...")

    while not queue.empty():
        i = queue.get() # get the ingredient to train

        print(f"[{rank}] ({i}) beginning training")

        # create the model
        this_model = tools.create_proper_model(
            graph, 
            model_to_use, 
            num_layers, 
            hidden_layer_size, 
            dropout, 
            aggregator_type, 
            num_heads, 
            input_drop, 
            attn_drop, 
            edge_drop, 
            use_attn_dst, 
            execute_silently=True
        )
        this_model.load_state_dict(initial_model.state_dict()) # load the initial model state dict (so that all models start from the same point)

        # train the model
        th.cuda.reset_peak_memory_stats()
        start_time = time.time()
        produce_training_output = False
        train_full_set_minibatched(graph, this_model, epochs, lr, wd, train_nids, batch_size, sampled_neighbors, print_progress=produce_training_output, device=device)
        max_memory = th.cuda.max_memory_allocated(device)
        end_time = time.time()
        print(f"[{rank}] ({i}) training took {round(end_time - start_time, 2)}s and {str(round(max_memory / 1000000, 2))} MB of memory")

        # test the model
        th.cuda.reset_peak_memory_stats()
        if test_fanout == -1 and test_batch_size == -1: # full batch testing
            test_acc = tools.test(graph, this_model, device=device).item()
        else: # testing with either neighbor sampling or batching, will need to use a dataloader
            test_acc = tools.batched_test(graph, this_model, test_fanout, test_batch_size, device=device, print_progress=False)
        max_memory = th.cuda.max_memory_allocated(device)
        print(f"[{rank}] ({i}) test accuracy: {round(test_acc * 100, 2)}% (took {str(round(max_memory / 1000000, 2))} MB of memory)")

        # save the model
        th.save(this_model.state_dict(), os.path.join(save_directory, f"ingredient_{i}.pt"))

def main(args):
    # distributed preparation
    mp.set_start_method('spawn')
    ingredient_queue = mp.Queue()

    # for each ingredient to train, add a corresponding number to the queue
    for i in range(args.starting_ingredient, args.num_to_train):
        ingredient_queue.put(i)

    # load the graph
    graph = tools.load_graph_dataset(args.graph_name)

    # set the initial model if it is none
    initial_model = tools.create_proper_model(
        graph, 
        args.model_to_use, 
        args.num_layers, 
        args.hidden_layer_size, 
        args.dropout, 
        args.aggregator_type, 
        args.num_heads, 
        args.input_drop, 
        args.attn_drop, 
        args.edge_drop, 
        args.use_attn_dst, 
        execute_silently=True
    )

    # create the save directory if it does not already exist
    if not os.path.exists(args.save_directory):
        os.makedirs(args.save_directory)

    # save the initial model
    th.save(initial_model.state_dict(), os.path.join(args.save_directory, "initial_model.pt"))

    # load the initial model state dict if we are continuing training
    if args.starting_ingredient > 0:
        initial_model.load_state_dict(th.load(os.path.join(args.save_directory, f"initial_model.pt")))

    # spawn the workers
    print(f"Spawning {args.num_workers} workers to train {args.num_to_train} ingredients...")
    print("Progress will be shown in the following format: \n[rank] (ingredient) message")
    for i in range(args.num_workers):
        p = mp.Process(
            target=ingredient_trainer_worker, 
            args=(
                i, 
                ingredient_queue,
                initial_model,
                args.save_directory, 
                graph, 
                args.epochs, 
                args.learn_rate, 
                args.weight_decay, 
                args.batch_size, 
                args.sampled_neighbors, 
                args.model_to_use, 
                args.hidden_layer_size, 
                args.dropout, 
                args.num_layers, 
                args.num_heads, 
                args.input_drop, 
                args.attn_drop, 
                args.edge_drop, 
                args.use_attn_dst, 
                args.aggregator_type, 
                args.test_batch_size, 
                args.test_fanout,
                args.full_graph_on_device
            )
        )
        p.start()
    
    # join the workers
    for i in range(args.num_workers):
        p.join()

if __name__ == '__main__':
    """
    Argument parsing
    """
    parser = argparse.ArgumentParser(description='Run souping experiments, including cross validation and testing')

    parser.add_argument('--num_workers', type=int, default=th.cuda.device_count(), help='The number of workers to spawn. This should be equal to the number of GPUs available')

    """
    Graph settings
    """
    parser.add_argument('--graph_name', type=str, required=True, help='The name of the graph to perform cross validation on')
    parser.add_argument('--full_graph_on_device', type=bool, default=False, 
                        help='Whether or not to keep the full graph on the device. This decreases time to train at the cost of GPU memory usage.')

    """
    Model Hyperparameters
    """
    parser.add_argument('--model_to_use', type=str, required=True,
                        help='the model to use for souping. Options: graphsage, gcn, or gat')
    parser.add_argument('--hidden_layer_size', type=int, 
                        help='the size of the hidden layers (GraphSAGE/GCN/GAT)')
    parser.add_argument('--dropout', type=float, 
                        help='the dropout to use for all non-input layers (GraphSAGE/GCN/GAT)')
    parser.add_argument('--num_layers', type=int, 
                        help='the number of layers to use (GraphSAGE/GCN/GAT)')
    """
    GAT Specific model hyperparameters
    """
    parser.add_argument('--num_heads', type=int, 
                        help='the number of heads to use (GAT)')
    parser.add_argument('--input_drop', type=float, 
                        help='the input dropout to use (GAT)')
    parser.add_argument('--attn_drop', type=float, 
                        help='the attention dropout to use (GAT)')
    parser.add_argument('--edge_drop', type=float, 
                        help='the edge dropout to use (GAT)')
    parser.add_argument('--use_attn_dst', type=bool,  default=False,
                        help='whether or not to use attention on the destination nodes (GAT)')
    """
    GraphSAGE Specific model hyperparameters
    """
    parser.add_argument('--aggregator_type', type=str,   default='mean', # most common aggregator
                        help='the aggregator type to use for all SageConv layers (GraphSAGE). Options: mean, gcn, lstm, and pool')
    
    """
    Batch training hyperparameters
    """
    parser.add_argument('--num_to_train', type=int,  default=1, # only train one model
                        help='the number of models to train')
    parser.add_argument('--starting_ingredient', type=int,  default=0, # start from the beginning
                        help='the ingredient to start from')
    parser.add_argument('--save_directory', type=str, required=True, help='The directory to save the models to')
    parser.add_argument('--batch_size', type=int,  default=-1, # full batch training
                        help='the batch size to use')
    parser.add_argument('--sampled_neighbors', type=int,  default=-1, # full neighborhood sampling
                        help='the number of sampled neighbors to use')
    parser.add_argument('--epochs', type=int,  required=True,
                        help='the number of epochs to use')
    parser.add_argument('--learn_rate', type=float,  required=True,
                        help='the learning rate to use')
    parser.add_argument('--weight_decay', type=float,  default=0.0,
                        help='the weight decay to use')
    
    """
    Testing hyperparameters
    """
    parser.add_argument('--test_batch_size', type=int, default=-1,
                        help='the batch size to use for testing')
    parser.add_argument('--test_fanout', type=int,  default=-1,
                        help='the fanout to use for testing')

    args = parser.parse_args()

    """
    Check that model hyperparameters are passed correctly
    """
    if args.model_to_use == 'graphsage':
        # ensure all hyperparams are passed for graphsage
        assert args.hidden_layer_size is not None
        assert args.dropout is not None
        assert args.aggregator_type is not None
        assert args.num_layers is not None

        # ensure no hyperparams are passed for gat
        #assert args.num_heads is None
        #assert args.input_drop is None
        #assert args.attn_drop is None
        #assert args.edge_drop is None
        #assert args.use_attn_dst is None
    elif args.model_to_use == 'gat':
        # ensure all hyperparams are passed for gat
        assert args.hidden_layer_size is not None
        assert args.dropout is not None
        assert args.num_layers is not None
        assert args.num_heads is not None
        assert args.input_drop is not None
        assert args.attn_drop is not None
        assert args.edge_drop is not None
        assert args.use_attn_dst is not None

        # ensure no hyperparams are passed for graphsage
        #assert args.aggregator_type is None
    elif args.model_to_use == 'gcn':
        # ensure all hyperparams are passed for gcn
        assert args.hidden_layer_size is not None
        assert args.dropout is not None
        assert args.num_layers is not None

        # ensure no hyperparams are passed for graphsage
        #assert args.aggregator_type is None

        # ensure no hyperparams are passed for gat
        #assert args.num_heads is None
        #assert args.input_drop is None
        #assert args.attn_drop is None
        #assert args.edge_drop is None
        #assert args.use_attn_dst is None     
    else:
        print("Invalid model type")
        raise NotImplementedError
    
    main(args)