-- Kenneth Marino
-- 06/2016
-- Graph testing Testing
require '../gsnn'

color = require 'trepl.colorize'

TOLERANCE = 1e-5
EPS = 1e-8

---------------------------- utils ---------------------------------

function print_test_name(test_name)
    print(color.blue(test_name))
end

function print_vector(v, v_name, precision)
    precision = precision or 8
    io.write(color.yellow(v_name) .. ': [')
    for i=1,v:nElement() do
        io.write(string.format('%' .. (precision + 4) .. '.' .. precision .. 'f', v[i]))
    end
    io.write(' ]\n')
end

function time_func(f, ...)
    local timer = torch.Timer()
    f(...)
    print(string.format('Time: %.2fs', timer:time().real))
    print('')
end

function create_sample_graph()
    local edges = {{{1,1,2}, {2,1,3}, {1,2,3}}, {{1,1,2}, {1,2,2}, {2,2,1}}}
    local annotations = {{{0,1}, {1,1}, {1,0}}, {{0,0}, {1,0}}}
    local n_edge_types = 2
    local n_total_nodes = #annotations[1] + #annotations[2]

    return edges, annotations, n_edge_types, n_total_nodes
end

--------------------- test cases ------------------------


-- Just for my own edifcation to figure out what the internals of ggnns looks like
function test_visualize_data_structs()
    local edges, annotations, n_edge_types, n_total_nodes = create_sample_graph()
    local n_steps = 3
    local annotation_dim = #annotations[1][1]
    local state_dim = 7

    local target = torch.randn(n_total_nodes, state_dim)
    local c = nn.MSECriterion()

    local net = ggnn.BaseGGNN(state_dim, annotation_dim, {}, n_edge_types)
    
    print(net.module_dict)
    net.module_dict['node-update-gate'].weight:fill(-1)
    net.module_dict['node-update-gate'].bias:zero()
    net.module_dict['reverse-prop-2-1'].weight:fill(0.2)
    net.module_dict['reverse-prop-2-1'].bias:zero()
    net.module_dict['prop-2-1'].weight:fill(1)
    net.module_dict['prop-2-1'].bias:zero()
    net.module_dict['node-update-transform'].weight:fill(0.1)
    net.module_dict['node-update-transform'].bias:zero()
    net.module_dict['reverse-prop-1-1'].weight:fill(-0.4)
    net.module_dict['reverse-prop-1-1'].bias:zero()
    net.module_dict['prop-1-1'].weight:fill(-0.3)
    net.module_dict['prop-1-1'].bias:zero()

    local w, dw = net:getParameters()
    dw:zero()

    local y = net:forward(edges, n_steps, annotations)
    c:forward(y, target)
    local dy = c:backward(y, target)
    net:backward(dy)

    -- Print out state variables of interest
    print(net.prop_inputs) 
    print(net.prop_inputs[1])

    -- Test it's the same if yuo give it an annotation tensor
    local annotation_tensor = torch.FloatTensor(n_total_nodes, annotation_dim)
    local node_count = 1
    for i = 1, #annotations do
        for j = 1, #annotations[i] do
            annotation_tensor[node_count][1] = annotations[i][j][1]
            annotation_tensor[node_count][2] = annotations[i][j][2]
            node_count = node_count + 1
        end 
    end
    print(annotation_tensor)
    local adjacency_matrix_list = {}
    for i = 1, #edges do
        adjacency_matrix_list[i] = ggnn.create_adjacency_matrix_cat(edges[i], #annotations[i], n_edge_types)
    end

    local net2 = ggnn.BaseGGNN(state_dim, annotation_dim, {}, n_edge_types)     
    net2.module_dict['node-update-gate'].weight:fill(-1)
    net2.module_dict['node-update-gate'].bias:zero()
    net2.module_dict['reverse-prop-2-1'].weight:fill(0.2)
    net2.module_dict['reverse-prop-2-1'].bias:zero()
    net2.module_dict['prop-2-1'].weight:fill(1)
    net2.module_dict['prop-2-1'].bias:zero()
    net2.module_dict['node-update-transform'].weight:fill(0.1)
    net2.module_dict['node-update-transform'].bias:zero()
    net2.module_dict['reverse-prop-1-1'].weight:fill(-0.4)
    net2.module_dict['reverse-prop-1-1'].bias:zero()
    net2.module_dict['prop-1-1'].weight:fill(-0.3)
    net2.module_dict['prop-1-1'].bias:zero()
 
    local y_test = net2:forward(adjacency_matrix_list, n_steps, annotation_tensor)
    print(y)
    print(y_test)

end

function test_gsnn_build()
    -- First Create a sample graph
    local graph1 = Graph(200, 200)
    graph1:addNewNode("fern", true, true, 33, 4)
    graph1:addNewNode("green", true, true, 12, 48)
    graph1:addNewNode("apple", true, true, 22, 93)
    graph1:addNewNode("round_shape", false, true, -1, 33)
    graph1:addNewNode("tomato", true, true, 41, 11)
    graph1:addNewNode("clock", true, true, 3, 49)
    graph1:addNewNode("wheel", true, true, 5, 5)
    graph1:addNewNode("tire", true, true, 93, 111)
    graph1:addNewNode("black", true, true, 8, 54)
    graph1:addNewNode("car", true, true, 74, 70)
    graph1:addNewNode("new_york", false, true, -1, 23)
    graph1:addNewNode("statue_of_liberty", false, true, -1, 80)
    graph1:addNewNode("raceway", true, true, 40, 99)
    graph1:addNewNode("taxi", true, true, 65, 20)
    graph1:addNewNode("car_dealership", false, true, -1, 55)
    graph1:addNewNode("street", true, true, 35, 81)
    graph1:addNewNode("snowy_weather", false, true, -1, 100)
    graph1:addEdge(1, "has", 1, 2, 0.9)
    graph1:addEdge(1, "has", 3, 2, 0.8)
    graph1:addEdge(2, "looks_like", 5, 3, 0.7)
    graph1:addEdge(2, "looks_like", 3, 5, 0.5)
    graph1:addEdge(1, "has", 3, 4, 1)
    graph1:addEdge(1, "has", 6, 4 ,0.89)
    graph1:addEdge(1, "has", 7, 4, 0.8)
    graph1:addEdge(1, "has", 8, 4, 0.77)
    graph1:addEdge(2, "looks_like", 7, 8, 0.8)
    graph1:addEdge(2, "looks_like", 8, 7, 0.98)
    graph1:addEdge(1, "has", 8, 9, 1)
    graph1:addEdge(3, "part_of", 7, 10, .9)
    graph1:addEdge(4, "found_in", 10, 11, 0.88)
    graph1:addEdge(1, "has", 11, 12, 0.58)
    graph1:addEdge(4, "found_in", 10, 13, 0.78)
    graph1:addEdge(4, "found_in", 14, 13, 0.86)
    graph1:addEdge(4, "found_in", 10, 15, 0.78)
    graph1:addEdge(4, "found_in", 10, 16, 0.78)
    graph1:addEdge(1, "has", 16, 17, 0.88)
    local detections = torch.FloatTensor(200):zero()
    detections[4] = 0.5
    detections[48] = 0.2
    detections[93] = 0.9
    detections[33] = 0.8
    detections[11] = 0.74
    detections[49] = 0.1
    detections[5] = 0.4
    detections[111] = 0.3
    detections[54] = 0.9
    detections[70] = 0.3
    detections[23] = 0.98
    detections[80] = 0.7
    detections[99] = 0.2
    detections[20] = 0.4
    detections[55] = 0.79
    detections[81] = 0.93
    detections[100] = 0.81
    local annotations_plus = torch.FloatTensor(200, 4):zero() 

    -- Initialize GSNN model
    params = {}
    local state_dim = 10
    local annotation_dim = 5
    local prop_net_h_sizes = {}
    local importance_out_net_h_sizes = {}
    local context_out_net_h_sizes = {}
    local n_edge_types = graph1.n_edge_types
    local num_steps = 3
    local context_dim = 10
    local num_expand = 3
    local init_conf = 0.8
    local n_total_nodes = graph1.n_total_nodes
    local node_bias_size = 2
    local num_inter_steps = 1
    local min_num_init = 3
    local context_net_options = {}
    context_net_options.architecture = "gated"
    context_net_options.transfer_function = "tanh"
    context_net_options.use_node_input = 1
    context_net_options.use_annotation_input = 1
    local importance_net_options = {}
    importance_net_options.architecture = "sigout"
    importance_net_options.transfer_function = "tanh"
    importance_net_options.use_node_input = 1
    importance_net_options.use_annotation_input = 1
    
    local net = gsnn.GSNN(state_dim, annotation_dim, prop_net_h_sizes, importance_out_net_h_sizes, context_out_net_h_sizes, n_edge_types, num_steps, min_num_init, context_dim, num_expand, init_conf, n_total_nodes, node_bias_size, num_inter_steps, context_net_options, importance_net_options) 

    print(net.state_dim)
    print(net.annotation_dim)
    print(net.prop_net_h_sizes)
    print(net.importance_out_net_h_sizes)
    print(net.context_out_net_h_sizes)
    print(net.n_edge_types)
    print(net.num_steps)
    print(net.min_num_init)
    print(net.context_dim)
    print(net.num_expand)
    print(net.init_conf)
    print(net.n_total_nodes)
    print(net.node_bias_size)
    print(net.num_inter_steps)
    print(net.context_architecture)
    print(net.context_transfer_function)
    print(net.context_use_node_input)
    print(net.context_use_annotation_input)
    print(net.importance_architecture)
    print(net.importance_transfer_function)
    print(net.importance_use_node_input)
    print(net.importance_use_annotation_input)
    print(net.importance_expand_type)
    --print(net.module_dict)

    local params, gradParams = net:getParameters()

    -- Foward prop
    local context_output, importance_outputs, reverse_lookup, active_idx, expanded_idx = net:forward(graph1, detections, annotations_plus)

    -- Backward prop
    local context_output_grad = torch.FloatTensor(context_output:size()):zero()
    local importance_output_grad = {}
    for i = 1, #importance_outputs do
        importance_output_grad[i] = torch.FloatTensor(importance_outputs[i]:size()):zero()
    end
    local context_input_grad, importance_input_grads, base_input_grads = net:backward(context_output_grad, importance_output_grad)

    -- Save and load as a new net
    gsnn.save_model_to_file('debug_model.t7', net, params)
    local copy = gsnn.load_gsnn_from_file('debug_model.t7')
    local paramsCopy, gradParamsCopy = copy:getParameters()

    -- Verify everything's the same
    assert(torch.eq(params, paramsCopy))
    assert(net.state_dim == copy.state_dim)
    assert(net.annotation_dim == copy.annotation_dim)
    for i = 1, #net.prop_net_h_sizes do
        assert(net.prop_net_h_sizes[i] == copy.prop_net_h_sizes[i])
    end
    for i = 1, #net.importance_out_net_h_sizes do
        assert(net.importance_out_net_h_sizes[i] == copy.importance_out_net_h_sizes[i])
    end
    for i = 1, #net.context_out_net_h_sizes do
        assert(net.context_out_net_h_sizes[i] == copy.context_out_net_h_sizes[i])
    end
    assert(net.n_edge_types == copy.n_edge_types)
    assert(net.num_steps == copy.num_steps)
    assert(net.min_num_init == copy.min_num_init)
    assert(net.context_dim == copy.context_dim)
    assert(net.num_expand == copy.num_expand)
    assert(net.init_conf == copy.init_conf)
    assert(net.n_total_nodes == copy.n_total_nodes)
    assert(net.node_bias_size == copy.node_bias_size)
    assert(net.num_inter_steps == copy.num_inter_steps)
    assert(net.context_architecture == copy.context_architecture)
    assert(net.context_transfer_function == copy.context_transfer_function)
    assert(net.context_use_node_input == copy.context_use_node_input)
    assert(net.context_use_annotation_input == copy.context_use_annotation_input)
    assert(net.importance_architecture == copy.importance_architecture)
    assert(net.importance_transfer_function == copy.importance_transfer_function)
    assert(net.importance_use_node_input == copy.importance_use_node_input)
    assert(net.importance_use_annotation_input == copy.importance_use_annotation_input)
    assert(net.importance_expand_type == copy.importance_expand_type)
end
-------------------- run tests -------------------------
time_func(test_visualize_data_structs)
time_func(test_gsnn_build)
