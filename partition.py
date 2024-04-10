import dgl
import torch as th
import json
import tqdm

"""
Globals and preloaded partition management
"""
global_partition_json_nmap = None
global_partition_json_nmap_keys = None
global_partition_json_nmap_loaded = False

#NOTE: these globals can be used to control if partitions are kept on each GPU (faster), or if they are kept on the CPU (more GPU memory efficient)
global_preloaded_partitions = False
global_keep_preloaded_partitions_on_gpu = False
global_preloaded_partitions_list = [] 

"""
Preloading partitions (e.g. moving them from disk to memory) can be done before distribution,
so that each worker does not have to load the partitions individually. This saves significant time.
"""
def preload_partitions(part_config, total_num_parts, device='cpu', progress_bar=False):
    global global_preloaded_partitions_list
    global global_preloaded_partitions

    global_preloaded_partitions_list = []

    if progress_bar:
        range_to_iterate = tqdm.tqdm(range(total_num_parts))
    else:
        range_to_iterate = range(total_num_parts)

    for partition_no in range_to_iterate:
        # load the partition
        part_structure, nfeats, efeats, pb, g_name, n_types, e_types = dgl.distributed.load_partition(part_config=part_config, part_id=partition_no)

        # remove/rename features from specific datasets that cause problems
        if '_N:_E:_N/feat' in efeats.keys():
            efeats.pop('_N:_E:_N/feat') # necessary for ogbn-proteins (not sure why)
        if '_N/species' in nfeats.keys():
            nfeats['_N/feat'] = nfeats['_N/species'] # necessary for ogbn-proteins (not sure why)
            nfeats.pop('_N/species')
        if '_N/label' in nfeats.keys():
            nfeats['_N/labels'] = nfeats['_N/label'] # necessary for reddit
            nfeats.pop('_N/label')
        if '_N:_E:_N/__orig__' in efeats.keys():
            efeats.pop('_N:_E:_N/__orig__') # necessary for reddit

        # move the partition to the target device
        if global_keep_preloaded_partitions_on_gpu:
            part_structure = part_structure.to(device)
            for feature_name in nfeats.keys():
                nfeats[feature_name] = nfeats[feature_name].to(device)
            for feature_name in efeats.keys():
                efeats[feature_name] = efeats[feature_name].to(device)
        else:
            part_structure = part_structure.to('cpu')
            for feature_name in nfeats.keys():
                nfeats[feature_name] = nfeats[feature_name].to('cpu')
            for feature_name in efeats.keys():
                efeats[feature_name] = efeats[feature_name].to('cpu')

        # append the partition to the list of partitions
        global_preloaded_partitions_list.append((part_structure, nfeats, efeats))
    global_preloaded_partitions = True

    return global_preloaded_partitions_list

def set_preloaded_partitions(preloaded_partitions, device='cpu'):
    global global_preloaded_partitions_list
    global global_preloaded_partitions
    global global_keep_preloaded_partitions_on_gpu

    global_preloaded_partitions_list = preloaded_partitions.copy()
    global_preloaded_partitions = True
    if device == 'cpu':
        global_keep_preloaded_partitions_on_gpu = False
    else:
        global_keep_preloaded_partitions_on_gpu = True

def free_preloaded_partitions():
    global global_preloaded_partitions_list
    global global_preloaded_partitions
    del global_preloaded_partitions_list
    global_preloaded_partitions = False


"""
Partition loading
"""
def load_partitions(part_config, total_num_parts, add_cut_edges, preassigned_partitions, device='cpu'):
    index_choices = [i for i in range(total_num_parts)]
    partition_to_load = preassigned_partitions[0]
    index_choices.remove(partition_to_load)

    from_local_part_to_global = {}
    if not global_preloaded_partitions:
            print("preloading partitions...")
            preload_partitions(part_config, total_num_parts, device=device, progress_bar=True)
    g_built, nfeats, efeats = global_preloaded_partitions_list[partition_to_load]
    
    # move the partition information to the target device
    g_built = g_built.to(device)
    if global_keep_preloaded_partitions_on_gpu:
        for feature_name in nfeats.keys():
            nfeats[feature_name] = nfeats[feature_name].to(device)
        for feature_name in efeats.keys():
            efeats[feature_name] = efeats[feature_name].to(device)
    else:
        for feature_name in nfeats.keys():
            nfeats[feature_name] = nfeats[feature_name].to('cpu')
        for feature_name in efeats.keys():
            efeats[feature_name] = efeats[feature_name].to('cpu')

    from_local_part_to_global[partition_to_load] = g_built.ndata[dgl.NID]
    cut_edges_mask = ~((g_built.ndata['part_id'][g_built.edges('uv')[1]] == partition_to_load))
    all_cut_u = from_local_part_to_global[partition_to_load][g_built.edges()[0][cut_edges_mask]]
    all_cut_v = from_local_part_to_global[partition_to_load][g_built.edges()[1][cut_edges_mask]]
    all_cut_u = all_cut_u.to(device)
    all_cut_v = all_cut_v.to(device)
    g_built = dgl.node_subgraph(
        g_built,
        g_built.ndata['inner_node'] == 1
    )
    g_built = g_built.to(device)
    for feature_name in nfeats.keys():
        g_built.ndata[feature_name] = nfeats[feature_name].to(device)
    # WARNING: edge data is not copied over!

    values_to_add_to_get_combined_ids = th.zeros(total_num_parts, dtype=th.int64).to(device)

    for i in range(1, len(preassigned_partitions)):
        partition_to_load = preassigned_partitions[i]
        index_choices.remove(partition_to_load)

        nodes_before_add = g_built.nodes().size()[0]
        part_structure, nfeats, efeats = global_preloaded_partitions_list[partition_to_load]

        # move the partition information to the target device
        part_structure.to(device)
        if global_keep_preloaded_partitions_on_gpu:
            part_structure = part_structure.to(device)
            for feature_name in nfeats.keys():
                nfeats[feature_name] = nfeats[feature_name].to(device)
            for feature_name in efeats.keys():
                efeats[feature_name] = efeats[feature_name].to(device)

        from_local_part_to_global[partition_to_load] = part_structure.ndata[dgl.NID]
        cut_edges_mask = ~((part_structure.ndata['part_id'][part_structure.edges('uv')[1]] == partition_to_load))
        these_global_cut_u = from_local_part_to_global[partition_to_load][part_structure.edges()[0][cut_edges_mask]]
        these_global_cut_v = from_local_part_to_global[partition_to_load][part_structure.edges()[1][cut_edges_mask]]

        u_to_add = these_global_cut_u
        v_to_add = these_global_cut_v
        u_to_add = u_to_add.to(device)
        v_to_add = v_to_add.to(device)
        
        all_cut_u = th.cat([all_cut_u.clone(), u_to_add])
        all_cut_v = th.cat([all_cut_v.clone(), v_to_add])

        only_internal_nodes = dgl.node_subgraph(
           part_structure,
           part_structure.ndata['inner_node'] == 1,
           relabel_nodes=True
        )
        only_internal_nodes = only_internal_nodes.to(device)
        for feature_name in only_internal_nodes.ndata.keys():
            only_internal_nodes.ndata[feature_name] = only_internal_nodes.ndata[feature_name].to(device)
        # WARNING: edge data is not copied over!


        for feature_name in nfeats.keys():
            only_internal_nodes.ndata[feature_name] = nfeats[feature_name].to(device)
        # WARNING: edge data is not copied over!

        values_to_add_to_get_combined_ids[partition_to_load] = nodes_before_add
        
        node_data_dict = { k:v for (k,v) in only_internal_nodes.ndata.items()}
        g_built = g_built.to(device)
        g_built.add_nodes(only_internal_nodes.nodes().size()[0], data=node_data_dict)
        g_built.add_edges(only_internal_nodes.edges()[0] + nodes_before_add, only_internal_nodes.edges()[1] + nodes_before_add) #NOTE: this does not add edge features or handle different edge types

    if add_cut_edges:
        # add all of the cut edges    
        u_pids = get_part_id_from_global_node_ids(all_cut_u, part_config, device=device)
        v_pids = get_part_id_from_global_node_ids(all_cut_v, part_config, device=device)
        u_locals = to_local_node_ids(all_cut_u, u_pids, part_config, device=device)
        v_locals = to_local_node_ids(all_cut_v, v_pids, part_config, device=device)
        for remaining_partition in index_choices:
            # remove the cut edges which go to the remaining partitions
            cut_edge_mask = v_pids != remaining_partition
            u_pids = u_pids[cut_edge_mask]
            v_pids = v_pids[cut_edge_mask]  
            u_locals = u_locals[cut_edge_mask]
            v_locals = v_locals[cut_edge_mask]
        u_combined = u_locals + values_to_add_to_get_combined_ids[u_pids]
        v_combined = v_locals + values_to_add_to_get_combined_ids[v_pids]
        g_built = dgl.add_edges(g_built, u_combined, v_combined)

    # convert the masks to boolean
    g_built.ndata['_N/train_mask'] = g_built.ndata['_N/train_mask'].bool()
    g_built.ndata['_N/val_mask'] = g_built.ndata['_N/val_mask'].bool()
    g_built.ndata['_N/test_mask'] = g_built.ndata['_N/test_mask'].bool()

    train_subgraph = dgl.node_subgraph(
        g_built,
        g_built.ndata['_N/train_mask']
    )
    train_nid = train_subgraph.ndata[dgl.NID]
    train_nid = train_nid.to(device)

    val_subgraph = dgl.node_subgraph(
        g_built,
        g_built.ndata['_N/val_mask']
    )
    val_nid = val_subgraph.ndata[dgl.NID]
    val_nid = val_nid.to(device)

    test_subgraph = dgl.node_subgraph(
        g_built,
        g_built.ndata['_N/test_mask']
    )
    test_nid = test_subgraph.ndata[dgl.NID]
    test_nid = test_nid.to(device)

    return g_built, train_nid, val_nid, test_nid

"""
Node ID management
"""
def to_global_node_ids(node_ids, this_partition, partition_json_filepath):
    global global_partition_json_nmap
    global global_partition_json_nmap_loaded
    global global_partition_json_nmap_keys

    # load the node mapping
    if not global_partition_json_nmap_loaded:
        partition_json = json.load(open(partition_json_filepath))
        global_partition_json_nmap = partition_json['node_map']
        global_partition_json_nmap_keys = list(global_partition_json_nmap.keys())
        global_partition_json_nmap_loaded = True

    assert len(global_partition_json_nmap_keys) == 1, "this function assumes the graph is homogeneous"

    for node_type in global_partition_json_nmap_keys:
        type_nmap = global_partition_json_nmap[node_type]
        partition_start, partition_end = type_nmap[this_partition]

    return node_ids + partition_start

def get_part_id_from_global_node_id(global_node_id, partition_json_filepath):
    global global_partition_json_nmap
    global global_partition_json_nmap_loaded
    global global_partition_json_nmap_keys

    # load the node mapping
    if not global_partition_json_nmap_loaded:
        partition_json = json.load(open(partition_json_filepath))
        global_partition_json_nmap = partition_json['node_map']
        global_partition_json_nmap_keys = list(global_partition_json_nmap.keys())
        global_partition_json_nmap_loaded = True
    
    assert len(global_partition_json_nmap_keys) == 1, "this function assumes the graph is homogeneous"

    for node_type in global_partition_json_nmap_keys:
        type_nmap = global_partition_json_nmap[node_type]
        for part_id in range(len(type_nmap)):
            partition_start, partition_end = type_nmap[part_id]
            if global_node_id >= partition_start and global_node_id < partition_end:
                return part_id
            
def get_part_id_from_global_node_ids(global_node_id, partition_json_filepath, device='cpu'):
    # tensor optimized version of get_part_id_from_global_node_id
    global global_partition_json_nmap
    global global_partition_json_nmap_loaded
    global global_partition_json_nmap_keys

    # load the node mapping
    if not global_partition_json_nmap_loaded:
        partition_json = json.load(open(partition_json_filepath))
        global_partition_json_nmap = partition_json['node_map']
        global_partition_json_nmap_keys = list(global_partition_json_nmap.keys())
        global_partition_json_nmap_loaded = True
    
    assert len(global_partition_json_nmap_keys) == 1, "this function assumes the graph is homogeneous"

    return_tensor = th.ones(global_node_id.size(), dtype=th.int32) * -1
    return_tensor = return_tensor.to(device)

    for node_type in global_partition_json_nmap_keys:
        type_nmap = global_partition_json_nmap[node_type]
        for part_id in range(len(type_nmap)):
            partition_start, partition_end = type_nmap[part_id]
            return_tensor = th.where((global_node_id >= partition_start) & (global_node_id < partition_end), part_id, return_tensor)

    assert th.sum(return_tensor == -1) == 0, "some global node ids were not found in the partition book"

    return return_tensor

def to_local_node_id(node_ids, this_partition, partition_json_filepath):
    global global_partition_json_nmap
    global global_partition_json_nmap_loaded
    global global_partition_json_nmap_keys

    # load the node mapping
    if not global_partition_json_nmap_loaded:
        partition_json = json.load(open(partition_json_filepath))
        global_partition_json_nmap = partition_json['node_map']
        global_partition_json_nmap_keys = list(global_partition_json_nmap.keys())
        global_partition_json_nmap_loaded = True

    assert len(global_partition_json_nmap_keys) == 1, "this function assumes the graph is homogeneous"

    for node_type in global_partition_json_nmap_keys:
        type_nmap = global_partition_json_nmap[node_type]
        partition_start, partition_end = type_nmap[this_partition]

    return node_ids - partition_start

def to_local_node_ids(node_ids, this_partition, partition_json_filepath, device='cpu'):
    # tensor optimized version of to_local_node_id (allows this_partition to be a tensor)
    global global_partition_json_nmap
    global global_partition_json_nmap_loaded
    global global_partition_json_nmap_keys

    # load the node mapping
    if not global_partition_json_nmap_loaded:
        partition_json = json.load(open(partition_json_filepath))
        global_partition_json_nmap = partition_json['node_map']
        global_partition_json_nmap_keys = list(global_partition_json_nmap.keys())
        global_partition_json_nmap_loaded = True

    assert len(global_partition_json_nmap_keys) == 1, "this function assumes the graph is homogeneous"

    for node_type in global_partition_json_nmap_keys:
        type_nmap = global_partition_json_nmap[node_type]
        type_tensor = th.tensor([type_nmap[i][0] for i in range(len(type_nmap))], dtype=th.int32).to(device)
        partition_start = type_tensor[this_partition]

    return node_ids - partition_start
