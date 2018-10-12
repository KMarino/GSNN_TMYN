-- Some utility functions.
--
-- Kenneth Marino, 07/2016
--

-- Convert annotation table to tensor
function gsnn.create_ann_tensor_from_table(annotations, n_nodes, annotation_dim)
    local annotation_tensor = torch.Tensor(n_nodes, annotation_dim):zero()
    for i = 1, #annotations do
        for j = 1, annotation_dim do
            annotation_tensor[i][j] = annotations[i][j]
        end
    end
    return annotation_tensor
end

-- Convert node indices to 1-hot representation
function gsnn.get_one_hot(active_idx, n_nodes)
    local node_onehot = torch.Tensor(#active_idx, n_nodes):zero()
    for i = 1, #active_idx do
        local real_idx = active_idx[i]
        node_onehot[i][real_idx] = 1
    end
    return node_onehot
end

-- Convert node indices to proper sparse input
function gsnn.get_sparse_rep(active_idx)
    local sparse_table = {}
    for i = 1, #active_idx do
        table.insert(sparse_table, {active_idx[i], 1})
    end
    return torch.Tensor(sparse_table)
end

-- Convert node indices to proper lookup table input
function gsnn.get_lookuptable_rep(active_idx)
    return torch.Tensor(active_idx)
end

-- net is a GSNN model
-- params is its parameter vector
function gsnn.save_model_to_file(file_name, net, params)
    local d = net:get_constructor_param_dict()
    d['params'] = params
    torch.save(file_name, d)
end

