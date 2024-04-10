import torch as th
import copy
import models
import partition as pt
from tqdm import tqdm
import numpy as np
import tools

def interpolate(g, state1, state2, model, granularity, device=th.device('cpu'), progress_bar=True):
    """
    taken directly from 
    https://github.com/VITA-Group/graph_ladling/blob/main/assets/README.md

    Interpolates between two models and returns the best validation accuracy.
    Used by greedy interpolation.
    """

    # move everything to the device
    model.to(device)
    g = g.to(device)
    state1 = {k: v.to(device) for k, v in state1.items()}
    state2 = {k: v.to(device) for k, v in state2.items()}

    # this is linear interpolation between two models
    alpha = np.linspace(0, 1, granularity)
    max_val,  loc = -1,  -1
    if progress_bar:
        print("Interpolating...")
        alpha = tqdm(alpha)
    for i in alpha:
        sd = {}
        for k in state1.keys():
            sd[k] = state1[k].clone() * i + state2[k].clone() * (1 - i)
        model.load_state_dict(sd)
        valid_acc = tools.validate(g, model, device=device)
        if valid_acc > max_val:
            max_val = valid_acc
            loc = i
    sd = {}
    for k in state1.keys():
        sd[k] = state1[k].clone() * loc + state2[k].clone() * (1 - loc)
    return max_val, loc, sd

def greedy_interpolated_soup(g, model_paths, model, granularity, device=th.device('cpu'), show_progress=True):
    """
    Standard greedy interpolation algorithm (from Graph Ladling)
    g: the entire graph (not partitioned)
    model_paths: a list containing paths to the models (model_x.pt files) to soup
    model: the model to use for souping. Its parameters are ignored, it is only used to get the architecture.
    granularity: the granularity to use for interpolation
    device: the device to use
    """
    models = []
    for model_directory in model_paths:
        models.append(copy.deepcopy(model))
        models[-1].load_state_dict(th.load(model_directory, map_location=device))


    # move everything to the device
    for model in models:
        model.to(device)

    # get the validation accuracy of each model
    if show_progress:
        print("Getting validation accuracy of each model...")
    vals = []
    indices = []
    if show_progress:
        for index, model in tqdm(enumerate(models)):
            vals.append(tools.validate(g, model, device=device))
            indices.append(index)
            print("Model " + str(index) + " has val accuracy " + str(vals[-1]))
    else:
        for index, model in enumerate(models):
            vals.append(tools.validate(g, model, device=device))
            indices.append(index)

    # order the models so that the best model is first
    if show_progress:
        print("Ordering models...")
    sorted_models = []
    indices = [x for _, x in sorted(zip(vals, indices), reverse=True)]
    for index in indices:
        sorted_models.append(models[index])
    models = sorted_models

    # this is greedy interpolation
    if show_progress:
        print("Running greedy interpolation...")
    model_copy = copy.deepcopy(model) #prevent overwriting for other experiments
    sd = models[0].state_dict()
    if show_progress:
        for i in tqdm(range(1, len(models))):
            max_val, loc, sd = interpolate(g, sd, models[i].state_dict(), model_copy, granularity=granularity, device=device)
            print("Interpolation at ", loc, " with val accuracy ", max_val)
    else:
        for i in range(1, len(models)):
            max_val, loc, sd = interpolate(g, sd, models[i].state_dict(), model_copy, granularity=granularity, device=device, progress_bar=False)
    return sd

def learned_soup(g, model_paths, model, epochs, learn_rate, weight_decay, init='uniform', CAW_T0=4, progress_bar=True, device=th.device('cpu')):
    """
    Learned souping algorithm

    g: the entire graph (not partitioned)
    model_paths: a list containing paths to the models (model_x.pt files) to soup
    model: the model to use for souping. Its parameters are ignored, it is only used to get the architecture.
    epochs: the number of epochs to train for
    learn_rate: the learning rate to use
    weight_decay: the weight decay to use
    device: the device to use
    """
    # load the state dicts into cpu memory
    sds = [th.load(cp, map_location=device) for cp in model_paths]

    # create an internal model (we will need to remove its parameters, which would be destructive to the original model)
    internal_model = copy.deepcopy(model)
    internal_model.disable_minibatching()
    internal_model = internal_model.to(device)

    test_model = copy.deepcopy(internal_model)
    test_model.disable_minibatching()
    test_model = test_model.to(device)

    _, names = models.make_functional(internal_model)

    # get information used for training
    features = tools.get_features(g)
    labels = tools.get_labels(g)
    val_mask = tools.get_val_mask(g)

    # create the alpha model, and wrap it around the internal model
    paramslist = [tuple(v.detach().requires_grad_().to(device) for _, v in sd.items()) for i, sd in enumerate(sds)]
    th.cuda.empty_cache()
    alpha_model = models.AlphaWrapper(paramslist, internal_model, names, device=device, init=init)

    # create the optimizer and loss function
    optimizer = th.optim.SGD(alpha_model.parameters(), lr=learn_rate, weight_decay=weight_decay)
    criterion = th.nn.CrossEntropyLoss()
    lr_scheduler = th.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CAW_T0)

    # move the data to the device
    g = g.to(device)
    features = features.to(device)
    labels = labels.to(device)
    val_mask = val_mask.to(device)
    alpha_model.to(device)

    # train the model
    for epoch in range(epochs):
        optimizer.zero_grad()

        logits = alpha_model(g, features)

        loss = criterion(logits[val_mask], labels[val_mask])
        
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        if epoch % 5 == 0 and progress_bar:
            print(f"epoch {epoch} loss: {loss}")

    # return the state dict
    return alpha_model.get_state_dict(device=device)

def pls(model_paths, model, epochs, learn_rate, weight_decay, init, CAW_T0, json_file, num_to_select, total_parts, preloaded_parts=None, progress_bar=True, device=th.device('cpu')):
    """
    Learned souping algorithm

    g: the entire graph (not partitioned)
    model_paths: a list containing paths to the models (model_x.pt files) to soup
    model: the model to use for souping. Its parameters are ignored, it is only used to get the architecture.
    epochs: the number of epochs to train for
    learn_rate: the learning rate to use
    weight_decay: the weight decay to use
    device: the device to use
    """
    # load the state dicts into cpu memory
    sds = [th.load(cp, map_location=device) for cp in model_paths]

    # create an internal model (we will need to remove its parameters, which would be destructive to the original model)
    internal_model = copy.deepcopy(model)
    internal_model.disable_minibatching()
    internal_model = internal_model.to(device)

    _, names = models.make_functional(internal_model)

    # create the alpha model, and wrap it around the internal model
    paramslist = [tuple(v.detach().requires_grad_().to(device) for _, v in sd.items()) for i, sd in enumerate(sds)]
    th.cuda.empty_cache()
    alpha_model = models.AlphaWrapper(paramslist, internal_model, names, device=device, init=init)

    # create the optimizer and loss function
    optimizer = th.optim.SGD(alpha_model.parameters(), lr=learn_rate, weight_decay=weight_decay)
    criterion = th.nn.CrossEntropyLoss()
    lr_scheduler = th.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CAW_T0)

    alpha_model.to(device)

    if preloaded_parts is None:
        preloaded_parts = pt.preload_partitions(json_file, total_parts, device=device)
    else:
        pt.set_preloaded_partitions(preloaded_parts, device=device)

    # train the model
    for epoch in range(epochs):
        partition_selection = list(np.random.choice(total_parts, num_to_select, replace=False))
        subgraph, train_nids, val_nids, test_nids = pt.load_partitions(
            json_file,
            total_parts,
            True,
            partition_selection,
            device=device
        )
        
        subgraph = subgraph.to(device)
        features = tools.get_features(subgraph)
        labels = tools.get_labels(subgraph)

        logits = alpha_model(subgraph, features)

        loss = criterion(logits[val_nids], labels[val_nids])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        subgraph = subgraph.cpu()

        if epoch % 5 == 0 and progress_bar:
            print(f"epoch {epoch} loss: {loss}")

    # return the state dict
    return alpha_model.get_state_dict(device=device)