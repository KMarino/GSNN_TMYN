-- Kenneth Marino
-- 07/2016
--
-- Graph Search Neural Network class

local GSNN = torch.class('gsnn.GSNN')

-- Should take in dimension information and number of steps. Other stuff comes when we actually do foward with the actual graph
function GSNN:__init(state_dim, annotation_dim, prop_net_h_sizes, importance_out_net_h_sizes, context_out_net_h_sizes, n_edge_types, num_steps, min_num_init, context_dim, num_expand, init_conf, n_total_nodes, node_bias_size, num_inter_steps, context_net_options, importance_net_options, module_dict)
    self.state_dim = state_dim
    self.annotation_dim = annotation_dim

    assert(self.state_dim >= self.annotation_dim, 'state_dim must be no less than annotation_dim')

    self.n_edge_types = n_edge_types
    self.module_dict = module_dict or {}
    self.prop_net_h_sizes = prop_net_h_sizes
    self.num_steps = num_steps
    self.importance_out_net_h_sizes = importance_out_net_h_sizes
    self.context_out_net_h_sizes = context_out_net_h_sizes
    self.min_num_init = min_num_init
    self.context_dim = context_dim
    self.num_expand = num_expand
    self.init_conf = init_conf
    self.n_total_nodes = n_total_nodes
    self.node_bias_size = node_bias_size
    self.num_inter_steps = num_inter_steps
    
    -- Output net variables
    self.context_architecture = context_net_options.architecture
    self.context_transfer_function = context_net_options.transfer_function
    self.context_use_node_input = context_net_options.use_node_input
    self.context_use_annotation_input = context_net_options.use_annotation_input
    self.importance_architecture = importance_net_options.architecture
    self.importance_transfer_function = importance_net_options.transfer_function
    self.importance_use_node_input = importance_net_options.use_node_input
    self.importance_use_annotation_input = importance_net_options.use_annotation_input
    self.importance_expand_type = importance_net_options.expand_type or "value" 
  
    -- Do some input checking
    if self.context_use_node_input == 1 and self.n_total_nodes <= 0 then
        error('Need number of nodes to have node biases')  
    end
    if self.importance_use_node_input == 1 and self.n_total_nodes <= 0 then
        error('Need number of nodes to have node biases')
    end

    -- Create base networks
    self:create_base_nets()
    self:create_output_nets()
end

-- Create the component base nets
function GSNN:create_base_nets()
    -- Create base GGNNs
    self.baseggnns = {}
    for step = 1, self.num_steps do 
        -- Create the base GGNNs that share weights with each other
        if step == 1 then
            self.baseggnns[step] = ggnn.BaseGGNN(self.state_dim, self.annotation_dim, self.prop_net_h_sizes, self.n_edge_types) 
            -- Add contents of base net module_dict to our module{_dict
            for k, v in pairs(self.baseggnns[step].module_dict) do
                self.module_dict[k] = v
            end
        else
            self.baseggnns[step] = ggnn.BaseGGNN(self.state_dim, self.state_dim, self.prop_net_h_sizes, self.n_edge_types, self.module_dict) 
        end
    end
end

-- Create the output nets
function GSNN:create_output_nets()
    -- Create node selection nets
    self.importance_out_nets = {}
    for step = 1, self.num_steps - 1 do
        local h_input = nn.Identity()()
        local in_dim = self.state_dim
        local joined_input = h_input
        local node_input
        local ann_input
       
        -- Go through all possible input combos
        if self.importance_use_node_input == 1 then
            node_input = nn.Identity()()
            local node_bias = ggnn.create_or_share('LookupTable', gsnn.NODE_BIAS_IMPORTANCE_PREFIX, self.module_dict, {self.n_total_nodes, self.node_bias_size})(node_input)
            joined_input = nn.JoinTable(1, 1)({joined_input, node_bias})
            in_dim = in_dim + self.node_bias_size
        end
        if self.importance_use_annotation_input == 1 then
            ann_input = nn.Identity()()
            joined_input = nn.JoinTable(1, 1)({joined_input, ann_input})
            in_dim = in_dim + self.annotation_dim
        end
        local layer_input = joined_input
 
        -- Add gate if needed
        local gate
        if self.importance_architecture == 'gated' or self.importance_architecture == 'gatedsig' then
            gate = nn.Sigmoid()(ggnn.create_or_share('Linear', gsnn.IMPORTANCE_GATE_NET_PREFIX, self.module_dict, {in_dim, 1})(joined_input))
        end 

        for i, h_dim in ipairs(self.importance_out_net_h_sizes) do
            -- Add linear and transfer layers
            layer_input = ggnn.create_or_share('Linear', gsnn.IMPORTANCE_NET_PREFIX .. '-' .. i, self.module_dict, {in_dim, h_dim})(layer_input)
            if self.importance_transfer_function == 'tanh' then
                layer_input = nn.Tanh()(layer_input)           
            elseif self.importance_transfer_function == 'sigmoid' then
                layer_input = nn.Sigmoid()(layer_input)
            elseif self.importance_transfer_function == 'relu' then
                layer_input = nn.ReLU()(layer_input)
            else
                error('Option ' .. self.importance_transfer_function .. ' not valid')
            end
            in_dim = h_dim    
        end

        local output = ggnn.create_or_share('Linear', gsnn.IMPORTANCE_NET_PREFIX .. '-' ..(#self.importance_out_net_h_sizes+1), self.module_dict, {in_dim, 1})(layer_input)

        local final_output
        if self.importance_architecture == 'gated' then
            final_output = nn.CMulTable()({output, gate})
        elseif self.importance_architecture == 'gatedsig' then
            pend_output = nn.CMulTable()({output, gate})
            final_output = nn.Sigmoid()(pend_output)
        elseif self.importance_architecture == 'linout' then
            final_output = output
        elseif self.importance_architecture == 'sigout' then
            final_output = nn.Sigmoid()(output)
        elseif self.importance_architecture == 'tanhout' then
            local output_p1 = nn.Tanh()(output)
            local output_p2 = nn.AddConstant(1, true)(output_p1)
            final_output = nn.MulConstant(0.5, true)(output_p2)
        else
            error('Option ' .. self.importance_architecture .. ' not valid')
        end

        -- Create final output net 
        if self.importance_use_node_input == 1 and self.importance_use_annotation_input == 1 then
            self.importance_out_nets[step] = nn.gModule({h_input, node_input, ann_input}, {final_output})
 
        elseif self.importance_use_node_input == 1 and self.importance_use_annotation_input == 0 then
            self.importance_out_nets[step] = nn.gModule({h_input, node_input}, {final_output})
        elseif self.importance_use_node_input == 0 and self.importance_use_annotation_input == 1 then
            self.importance_out_nets[step] = nn.gModule({h_input, ann_input}, {final_output})
        else
            self.importance_out_nets[step] = nn.gModule({h_input}, {final_output})
        end
    end

    local h_input = nn.Identity()()
    local in_dim = self.state_dim
    local joined_input = h_input
    local node_input
    local ann_input
    
    -- Go through all possible input combos
    if self.context_use_node_input == 1 then
        node_input = nn.Identity()()
        local node_bias = ggnn.create_or_share('LookupTable', gsnn.NODE_BIAS_CONTEXT_PREFIX, self.module_dict, {self.n_total_nodes, self.node_bias_size})(node_input)
        joined_input = nn.JoinTable(1, 1)({joined_input, node_bias})
        in_dim = in_dim + self.node_bias_size
    end
    if self.context_use_annotation_input == 1 then
        ann_input = nn.Identity()()
        joined_input = nn.JoinTable(1, 1)({joined_input, ann_input})
        in_dim = in_dim + self.annotation_dim
    end
    local layer_input = joined_input
 
    -- Add gate if needed
    local gate
    if self.context_architecture == 'gated' then
        gate = nn.Sigmoid()(ggnn.create_or_share('Linear', gsnn.CONTEXT_GATE_NET_PREFIX, self.module_dict, {in_dim, self.context_dim})(joined_input))
    end 
    
    -- Add linear and transfer layers
    for i, h_dim in ipairs(self.context_out_net_h_sizes) do 
        layer_input = ggnn.create_or_share('Linear', gsnn.CONTEXT_NET_PREFIX .. '-' .. i, self.module_dict, {in_dim, h_dim})(layer_input)
        if self.context_transfer_function == 'tanh' then
            layer_input = nn.Tanh()(layer_input)           
        elseif self.context_transfer_function == 'sigmoid' then
            layer_input = nn.Sigmoid()(layer_input)
        elseif self.context_transfer_function == 'relu' then
            layer_input = nn.ReLU()(layer_input)
        else
            error('Option ' .. self.context_transfer_function .. ' not valid')
        end
        in_dim = h_dim
    end
    
    -- Final output layer
    local output = ggnn.create_or_share('Linear', gsnn.CONTEXT_NET_PREFIX .. '-' .. (#self.context_out_net_h_sizes+1), self.module_dict, {in_dim, self.context_dim})(layer_input)
    local final_output
    if self.context_architecture == 'gated' then
        final_output = nn.CMulTable()({output, gate})
    elseif self.context_architecture == 'linout' then
        final_output = output
    elseif self.context_architecture == 'sigout' then
        final_output = nn.Sigmoid()(output)
    elseif self.context_architecture == 'tanhout' then
        local output_p1 = nn.Tanh()(output)
        local output_p2 = nn.AddConstant(1, true)(output_p1)
        final_output = nn.MulConstant(0.5, true)(output_p2)
    else
        error('Option ' .. self.context_architecture .. ' not valid')
    end

    -- Create final output net
    if self.context_use_node_input == 1 and self.context_use_annotation_input == 1 then
        self.context_out_net = nn.gModule({h_input, node_input, ann_input}, {final_output})
    elseif self.context_use_node_input == 1 and self.context_use_annotation_input == 0 then
        self.context_out_net = nn.gModule({h_input, node_input}, {final_output})
    elseif self.context_use_node_input == 0 and self.context_use_annotation_input == 1 then
        self.context_out_net = nn.gModule({h_input, ann_input}, {final_output})
    else
        self.context_out_net = nn.gModule({h_input}, {final_output})
    end
end

-- Transfer context and importance net options to structures
function GSNN:getOutputNetStructs()
    -- Make data structures
    local context_net_options = {}
    local importance_net_options = {}
    context_net_options.architecture = self.context_architecture
    context_net_options.transfer_function = self.context_transfer_function
    context_net_options.use_node_input = self.context_use_node_input
    context_net_options.use_annotation_input = self.context_use_annotation_input
    importance_net_options.architecture = self.importance_architecture 
    importance_net_options.transfer_function = self.importance_transfer_function
    importance_net_options.use_node_input = self.importance_use_node_input 
    importance_net_options.use_annotation_input = self.importance_use_annotation_input 
    importance_net_options.expand_type = self.importance_expand_type 
 
    return context_net_options, importance_net_options
end

-- Creates a copy of the network sharing the same module_dict (same parameters)
function GSNN:create_share_param_copy()
    local context_net_options, importance_net_options = self:getOutputNetStructs()    
    return gsnn.GSNN(self.state_dim, self.annotation_dim, self.prop_net_h_sizes, self.importance_out_net_h_sizes, self.context_out_net_h_sizes, self.n_edge_types, self.num_steps, self.min_num_init, self.context_dim, self.num_expand, self.init_conf, self.n_total_nodes, self.node_bias_size, self.num_inter_steps, context_net_options, importance_net_options, self.module_dict)
end

-- Return a dictionary of parameters that can be used to call the constructor
-- to create a GSNN model with the same architecture. Used when saving a 
-- model to file
function GSNN:get_constructor_param_dict()
    local tmp_context_net_options, tmp_importance_net_options = self:getOutputNetStructs()     
    return {
        state_dim = self.state_dim,
        annotation_dim = self.annotation_dim,
        prop_net_h_sizes = self.prop_net_h_sizes,
        importance_out_net_h_sizes = self.importance_out_net_h_sizes,
        context_out_net_h_sizes = self.context_out_net_h_sizes,
        n_edge_types = self.n_edge_types,
        num_steps = self.num_steps,
        min_num_init = self.min_num_init,
        context_dim = self.context_dim,
        num_expand = self.num_expand,
        init_conf = self.init_conf,
        n_total_nodes = self.n_total_nodes,
        node_bias_size = self.node_bias_size,
        num_inter_steps = self.num_inter_steps,
        context_net_options = tmp_context_net_options,
        importance_net_options = tmp_importance_net_options
    }
end

-- Forward pass through GSNN 
function GSNN:forward(full_graph, initial_detections, initial_annotations)
    -- Get the initial detections
    local annotations, reverse_lookup, active_idx, expanded_idx, edges, edge_conf = full_graph:getInitialGraph(initial_detections, initial_annotations, self.annotation_dim, self.init_conf, self.min_num_init)
    
    -- Convert everything to tensors
    local adjacency_matrix_list = {}
    adjacency_matrix_list[1] = ggnn.create_adjacency_matrix_cat(edges, #active_idx, self.n_edge_types)
    local annotation_tensor = gsnn.create_ann_tensor_from_table(annotations, #annotations, self.annotation_dim) 
    local input_tensor = torch.Tensor(#active_idx, self.annotation_dim):zero()
    input_tensor:narrow(1, 1, #annotations):copy(annotation_tensor)

    -- State values
    self.past_active_idx = {}
    local initial_active_idx = {}
    
    for aidx = 1, #active_idx do
        table.insert(initial_active_idx, active_idx[aidx])
    end
    table.insert(self.past_active_idx, initial_active_idx)
    self.past_expanded_idx = {}
    local initial_expanded_idx = {}
    for eidx = 1, #expanded_idx do
        table.insert(initial_expanded_idx, expanded_idx[eidx])
    end
    table.insert(self.past_expanded_idx, initial_expanded_idx)

    self.base_net_inputs = {}
    self.base_net_outputs = {}
    self.importance_outputs = {}
    self.base_num_nodes = {}
    self.node_inputs = {}
    self.ann_inputs = {}
    table.insert(self.base_net_inputs, input_tensor)
    table.insert(self.base_num_nodes, #active_idx)

    for i = 1, self.num_steps do
        -- Forward through base ggnns
        local debug_out = self.baseggnns[i]:forward(adjacency_matrix_list, self.num_inter_steps, self.base_net_inputs[i])

        -- Add output of net to net_outputs
        table.insert(self.base_net_outputs, self.baseggnns[i].prop_inputs[self.num_inter_steps+1])

        if i ~= self.num_steps then
            -- Get annotation input
            local zero_pad_ann_input = torch.Tensor(#active_idx, self.annotation_dim):zero()
            zero_pad_ann_input:narrow(1, 1, #annotations):copy(annotation_tensor)
            self.ann_inputs[i] = zero_pad_ann_input

            -- Do importance calculation
            self.node_inputs[i] = gsnn.get_lookuptable_rep(active_idx)
            local importance 
            if self.importance_use_node_input == 1 and self.importance_use_annotation_input == 1 then
                importance = self.importance_out_nets[i]:forward({self.base_net_outputs[i], self.node_inputs[i], self.ann_inputs[i]})
 
            elseif self.importance_use_node_input == 1 and self.importance_use_annotation_input == 0 then
                importance = self.importance_out_nets[i]:forward({self.base_net_outputs[i], self.node_inputs[i]})
  
            elseif self.importance_use_node_input == 0 and self.importance_use_annotation_input == 1 then
                importance = self.importance_out_nets[i]:forward({self.base_net_outputs[i], self.ann_inputs[i]}) 
            else
                importance = self.importance_out_nets[i]:forward(self.base_net_outputs[i])
            end
            table.insert(self.importance_outputs, importance)

            -- Update graph imformation
            if self.importance_expand_type == "value" then 
                reverse_lookup, active_idx, expanded_idx, edges, edge_conf = full_graph:updateGraphFromImportance(torch.reshape(importance, importance:size(1)), reverse_lookup, active_idx, expanded_idx, edges, edge_conf, self.num_expand)
            elseif self.importance_expand_type == "select" then
                 reverse_lookup, active_idx, expanded_idx, edges, edge_conf = full_graph:updateGraphFromImportanceSelection(torch.reshape(importance, importance:size(1)), reverse_lookup, active_idx, expanded_idx, edges, edge_conf)
            end

            -- Update in state
            local new_active_idx = {}
            for aidx = 1, #active_idx do
                table.insert(new_active_idx, active_idx[aidx])
            end
            table.insert(self.past_active_idx, new_active_idx)
            local new_expanded_idx = {}
            for eidx = 1, #expanded_idx do
                table.insert(new_expanded_idx, expanded_idx[eidx])
            end
            table.insert(self.past_expanded_idx, new_expanded_idx)

            -- Update tensors and save everything
            adjacency_matrix_list = {}
            adjacency_matrix_list[1] = ggnn.create_adjacency_matrix_cat(edges, #active_idx, self.n_edge_types) 
            local next_input_tensor = torch.Tensor(#active_idx, self.state_dim):zero()
            next_input_tensor:narrow(1, 1, self.base_num_nodes[i]):copy(self.base_net_outputs[i])
            self.base_net_inputs[i+1] = next_input_tensor
            self.base_num_nodes[i+1] = #active_idx 
        end       
    end

    -- Get context output
    self.node_inputs[self.num_steps] = gsnn.get_lookuptable_rep(active_idx)
    local zero_pad_ann_input = torch.Tensor(#active_idx, self.annotation_dim):zero()
    zero_pad_ann_input:narrow(1, 1, #annotations):copy(annotation_tensor)
    self.ann_inputs[self.num_steps] = zero_pad_ann_input
    
    if self.context_use_node_input == 1 and self.context_use_annotation_input == 1 then
        self.context_output = self.context_out_net:forward({self.base_net_outputs[self.num_steps], self.node_inputs[self.num_steps], self.ann_inputs[self.num_steps]})
    elseif self.context_use_node_input == 1 and self.context_use_annotation_input == 0 then
        self.context_output = self.context_out_net:forward({self.base_net_outputs[self.num_steps], self.node_inputs[self.num_steps]})
    elseif self.context_use_node_input == 0 and self.context_use_annotation_input == 1 then
        self.context_output = self.context_out_net:forward({self.base_net_outputs[self.num_steps], self.ann_inputs[self.num_steps]})
    else
        self.context_output = self.context_out_net:forward({self.base_net_outputs[self.num_steps]})
    end

    -- Save some values we might want on hand
    self.reverse_lookup = reverse_lookup
    self.active_idx = active_idx
    self.expanded_idx = expanded_idx

    return self.context_output, self.importance_outputs, reverse_lookup, active_idx, expanded_idx
end

-- Backward pass through GSNN
function GSNN:backward(context_output_grad, importance_output_grad)
    -- Initialize base grads
    self.base_input_grads = {}
    self.base_output_grads = {}
    for i = 1, self.num_steps do
        local zero_input_grad = torch.Tensor(self.base_net_inputs[i]:size()):zero()
        local zero_output_grad = torch.Tensor(self.base_net_outputs[i]:size()):zero()
        table.insert(self.base_input_grads, zero_input_grad)
        table.insert(self.base_output_grads, zero_output_grad)
    end

    -- Backward through context net
    if self.context_use_node_input == 1 and self.context_use_annotation_input == 1 then
        self.context_input_grad = self.context_out_net:backward({self.base_net_outputs[self.num_steps], self.node_inputs[self.num_steps], self.ann_inputs[self.num_steps]}, context_output_grad)
        self.base_output_grads[self.num_steps]:add(self.context_input_grad[1]) 
    elseif self.context_use_node_input == 1 and self.context_use_annotation_input == 0 then
        self.context_input_grad = self.context_out_net:backward({self.base_net_outputs[self.num_steps], self.node_inputs[self.num_steps]}, context_output_grad)
        self.base_output_grads[self.num_steps]:add(self.context_input_grad[1]) 
    elseif self.context_use_node_input == 0 and self.context_use_annotation_input == 1 then
        self.context_input_grad = self.context_out_net:backward({self.base_net_outputs[self.num_steps], self.ann_inputs[self.num_steps]}, context_output_grad)
        self.base_output_grads[self.num_steps]:add(self.context_input_grad[1]) 
    else
        self.context_input_grad = self.context_out_net:backward(self.base_net_outputs[self.num_steps], context_output_grad)
        self.base_output_grads[self.num_steps]:add(self.context_input_grad) 
    end
    
    -- Backward through importance nets
    self.importance_input_grads = {}
    for i = 1, self.num_steps-1 do
        if self.importance_use_node_input == 1 and self.importance_use_annotation_input == 1 then      
            self.importance_input_grads[i] = self.importance_out_nets[i]:backward({self.base_net_outputs[i], self.node_inputs[i], self.ann_inputs[i]}, importance_output_grad[i])
            self.base_output_grads[i]:add(self.importance_input_grads[i][1])
        elseif self.importance_use_node_input == 1 and self.importance_use_annotation_input == 0 then  
            self.importance_input_grads[i] = self.importance_out_nets[i]:backward({self.base_net_outputs[i], self.node_inputs[i]}, importance_output_grad[i])
            self.base_output_grads[i]:add(self.importance_input_grads[i][1]) 
        elseif self.importance_use_node_input == 0 and self.importance_use_annotation_input == 1 then 
            self.importance_input_grads[i] = self.importance_out_nets[i]:backward({self.base_net_outputs[i], self.ann_inputs[i]}, importance_output_grad[i])
            self.base_output_grads[i]:add(self.importance_input_grads[i][1])
        else
            self.importance_input_grads[i] = self.importance_out_nets[i]:backward(self.base_net_outputs[i], importance_output_grad[i])
            self.base_output_grads[i]:add(self.importance_input_grads[i])
        end
    end

    -- Backward through the base nets 
    for i = self.num_steps, 1, -1 do
        -- Backward through base net with base output grad
        self.base_input_grads[i] = self.baseggnns[i]:backward(self.base_output_grads[i])
       
        -- Update base_output_grad of previous base net
        if i ~= 1 then
            self.base_output_grads[i-1]:add(torch.Tensor(self.base_output_grads[i-1]:size()):copy(self.base_input_grads[i]:narrow(1, 1, self.base_num_nodes[i-1])))
        end
    end
    
    return self.context_input_grad, self.importance_input_grads, self.base_input_grads 
end

-- Load a GSNN model from file, this must be paired with get_constructor_params_dict.
function gsnn.load_gsnn_from_file(file_name)
    local d = torch.load(file_name)
    local net = gsnn.GSNN(d['state_dim'], d['annotation_dim'], d['prop_net_h_sizes'], d['importance_out_net_h_sizes'], d['context_out_net_h_sizes'], d['n_edge_types'], d['num_steps'], d['min_num_init'], d['context_dim'], d['num_expand'], d['init_conf'], d['n_total_nodes'], d['node_bias_size'], d['num_inter_steps'], d['context_net_options'], d['importance_net_options'])
    local w = net:getParameters()
    w:copy(d['params'])

    return net
end

-- Return a list of parameters and a list of parameter gradients
-- Code vopied from BaseGGNN.lua
function GSNN:parameters()
    local w = {}
    local dw = {}

    -- sort the keys to make sure the parameters are always in the same order
    local k_list = {}
    for k, v in pairs(self.module_dict) do
        table.insert(k_list, k)
    end
    table.sort(k_list)

    for i=1,#k_list do
        m = self.module_dict[k_list[i]]
        local mw, mdw = m:parameters()
        if mw then
            if type(mw) == 'table' then
                for i=1,#mw do
                    table.insert(w, mw[i])
                    table.insert(dw, mdw[i])
                end
            else
                table.insert(w, mw)
                table.insert(dw, mdw)
            end
        end
    end
    return w, dw
end

-- Return a flattened version of the list of parameters, and a flattened version
-- of the list of parameter gradients.
--
-- This function directly calls nn.Module.flatten
-- This function also copied from BaseGGNN.lua
function GSNN:getParameters()
    local params, grad_params = self:parameters()
    return nn.Module.flatten(params), nn.Module.flatten(grad_params)
end
