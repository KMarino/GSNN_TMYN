-- Kenneth Marino
-- 06/2016
-- Graph testing Testing

require '../graph'

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
    --io.read()
end

--------------------- test cases ------------------------

function test_node_create()
    -- Create a couple nodes
    local node1 = Node(1, "Cat", true, true, 120, 4)
    print(node1)
    local node2 = Node(2, "Dog", true, true, 121, 2)
    print(node2)
    local node3 = Node(3, "Dog_Hat", false, true, -1, 100)
    print(node3)
    local node4 = Node(4, "Blah", false, false, -1, -1)
    print(node4)
end

function test_edge_create()
    -- Create a couple nodes
    local node1 = Node(1, "Cat", true, true, 120, 4)
    local node2 = Node(2, "Dog", true, true, 121, 2)
    local node3 = Node(3, "Dog_Hat", false, true, -1, 100)

    -- Create a couple edge types
    local edgetype1 = EdgeType(1, "is_on")
    print(edgetype1)
    local edgetype2 = EdgeType(2, "looks_like")
    print(edgetype2)
    
    -- Create a couple edges
    local edge1 = Edge(edgetype2, node1, node2, 1, 0.9)
    print(edge1)
    local edge2 = Edge(edgetype1, node3, node2, 2, 0.8)
    print(edge2)

    -- Make sure references work
    node1.index = 4
    node2.index = 5
    node3.index = 6
    print(edge1)
    print(edge2)    
end

function test_graph_create()
    -- Start with an empty graph
    local graph1 = Graph(100, 100)
    print(graph1)

    -- Add some nodes
    graph1:addNewNode("Person", true, true, 33, 4)
    graph1:addNewNode("Cat", true, true, 54, 2)
    graph1:addNewNode("Dog", true, true, 20, 100)
    graph1:addNewNode("Hospital", true, false, 43, -1)
    graph1:addNewNode("Big_Dog", false, false, -1, -1)
    print(graph1)

    -- Add edges
    graph1:addEdge(1, "looks_like", 2, 3, 0.9)
    graph1:addEdge(2, "is", 5, 3, 0.8)
    graph1:addEdge(1, "looks_like", 1, 3, 0.85)
    print(graph1)
    print(graph1.nodes[1].outgoing_edges)
    print(graph1.nodes[1].incoming_edges)
    print(graph1.nodes[2].outgoing_edges)
    print(graph1.nodes[2].incoming_edges)
    print(graph1.nodes[3].outgoing_edges)
    print(graph1.nodes[3].incoming_edges)
    print(graph1.nodes[4].outgoing_edges)
    print(graph1.nodes[4].incoming_edges)
    print(graph1.nodes[5].outgoing_edges)
    print(graph1.nodes[5].incoming_edges)
end

function test_getexpanded()
    -- Get graph
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
    print(graph1)
 
    -- Test getExpandedGraph
    local reverse_lookup = torch.IntTensor(17):fill(-1)
    local active_idx = {4, 9, 10, 1}
    reverse_lookup[4] = 1
    reverse_lookup[9] = 2
    reverse_lookup[10] = 3
    reverse_lookup[1] = 4
    local expand_idx = {4, 9, 10}
    edges = {}
    edge_conf = {}
    reverse_lookup, active_idx, edges, edge_conf = graph1:getExpandedGraph(reverse_lookup, active_idx, expand_idx, edges, edge_conf) 
    print('***')
    print(reverse_lookup)
    print(active_idx)
    print(edges)
    print(edge_conf)

    -- Test, but expand some more nodes
    expand_idx = {3, 13, 16}
    reverse_lookup, active_idx, edges, edge_conf = graph1:getExpandedGraph(reverse_lookup, active_idx, expand_idx, edges, edge_conf) 
    print(reverse_lookup)
    print(active_idx)
    print(edges)
    print(edge_conf)
end

function test_getexpandedwa()
    -- Get graph
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
    print(graph1)
 
    -- Test getExpandedGraph
    local reverse_lookup = torch.IntTensor(17):fill(-1)
    local active_idx = {4, 9, 10, 1}
    reverse_lookup[4] = 1
    reverse_lookup[9] = 2
    reverse_lookup[10] = 3
    reverse_lookup[1] = 4
    local expand_idx = {4, 9, 10}
    edges_interior = {}
    edges_frontier = {}
    reverse_lookup, active_idx, edges_interior, edges_frontier = graph1:getExpandedGraphWA(reverse_lookup, active_idx, expand_idx, edges_interior, edges_frontier) 
    print('XXX')
    print(reverse_lookup)
    print(active_idx)
    print(edges_interior)
    print(edges_frontier)

    -- Test, but expand some more nodes
    expand_idx = {3, 13, 16}
    reverse_lookup, active_idx, edges_interior, edges_frontier = graph1:getExpandedGraphWA(reverse_lookup, active_idx, expand_idx, edges_interior, edges_frontier) 
    print(reverse_lookup)
    print(active_idx)
    print(edges_interior)
    print(edges_frontier)
end

function test_get_initial_graph()
    torch.setdefaulttensortype('torch.FloatTensor')

    -- Get graph
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
    print(graph1)

    local min_num = 3
    local conf_thresh = 0.8
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
    local ann_total_size = 5

    local detect_conf, reverse_lookup, active_idx, expanded_idx, edges, edge_conf = graph1:getInitialGraph(detections, annotations_plus, ann_total_size, conf_thresh, min_num)
    print('11111')
    print(detect_conf)
    print(reverse_lookup)
    print(active_idx)
    print(expanded_idx)
    print(edges)
    print(edge_conf)

    -- Do it again, but with not enough detections
    detections:zero()
    detections[20] = 0.82
    detections[33] = 0.4
    detections[49] = 0.99

    detect_conf, reverse_lookup, active_idx, expanded_idx, edges, edge_conf = graph1:getInitialGraph(detections, annotations_plus, ann_total_size, conf_thresh, min_num)
    print(detect_conf)
    print(reverse_lookup)
    print(active_idx)
    print(expanded_idx)
    print(edges)
    print(edge_conf)
end

function test_get_initial_graph_wa()
    torch.setdefaulttensortype('torch.FloatTensor')

    -- Get graph
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
    print(graph1)

    local min_num = 3
    local conf_thresh = 0.8
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
    local ann_total_size = 5
    local detect_conf, reverse_lookup, active_idx, expanded_idx, edges_interior, edges_frontier = graph1:getInitialGraphWA(detections, annotations_plus, ann_total_size, conf_thresh, min_num)
    print('22222')
    print(detect_conf)
    print(reverse_lookup)
    print(active_idx)
    print(expanded_idx)
    print(edges_interior)
    print(edges_frontier)

    -- Do it again, but with not enough detections
    detections:zero()
    detections[20] = 0.82
    detections[33] = 0.4
    detections[49] = 0.99

    detect_conf, reverse_lookup, active_idx, expanded_idx, edges_interior, edges_frontier = graph1:getInitialGraphWA(detections, annotations_plus, ann_total_size, conf_thresh, min_num)
    print(detect_conf)
    print(reverse_lookup)
    print(active_idx)
    print(expanded_idx)
    print(edges_interior)
    print(edges_frontier)
end

function test_update_graph_from_importance()
    torch.setdefaulttensortype('torch.FloatTensor')

    -- Get graph
    local graph1 = Graph(100, 100)
    graph1:addNewNode("fish", true, true, 42, 13)
    graph1:addNewNode("shark", true, true, 39, 80)
    graph1:addNewNode("aquarium", true, true, 22, 74)
    graph1:addNewNode("blue", true, true, 83, 32)
    graph1:addNewNode("water", true, true, 65, 2)
    graph1:addNewNode("beverage", true, false, 11, -1)
    graph1:addNewNode("milk", true, true, 40, 10)
    graph1:addNewNode("food", true, false, 59, -1)
    graph1:addNewNode("plate", true, true, 38, 24)
    graph1:addNewNode("round_shape", false, true, -1, 9)
    graph1:addNewNode("apple", true, true, 21, 93)
    graph1:addNewNode("orange", true, true, 72, 44)
    graph1:addNewNode("fruit", true, false, 91, -1)
    graph1:addNewNode("green", true, true, 12, 48)
    graph1:addNewNode("grass", true, true, 2, 8)
    graph1:addNewNode("lawn", true, true, 49, 18)
    graph1:addNewNode("house", true, true, 100, 1)
    graph1:addEdge(1, "looks_like", 1, 2, 0.9)
    graph1:addEdge(1, "looks_like", 2, 1, 0.8)
    graph1:addEdge(2, "found_in", 1, 3, 0.87)
    graph1:addEdge(2, "found_in", 2, 3, 0.92)
    graph1:addEdge(3, "has", 3, 4, 0.7)
    graph1:addEdge(4, "part_of", 5, 3, 0.67)
    graph1:addEdge(3, "has", 5, 4, 0.54)
    graph1:addEdge(5, "is", 5, 6, 0.68)
    graph1:addEdge(5, "is", 7, 6, 0.88)
    graph1:addEdge(5, "is", 6, 8, 0.83)
    graph1:addEdge(2, "found_in", 8, 9, 0.99)
    graph1:addEdge(5, "is", 11, 8, 1)
    graph1:addEdge(5, "is", 13, 8, 0.44)
    graph1:addEdge(5, "is", 12, 8, 0.73)
    graph1:addEdge(3, "has", 9, 10, 0.93)
    graph1:addEdge(3, "has", 11, 10, 0.54)
    graph1:addEdge(5, "is", 11, 13, 0.76)
    graph1:addEdge(3, "has", 11, 14, 0.36)
    graph1:addEdge(5, "is", 12, 13, 0.9)
    graph1:addEdge(3, "has", 15, 14, 0.8)
    graph1:addEdge(4, "part_of", 15, 16, 0.7)
    graph1:addEdge(4, "part_of", 17, 16, 0.9)
    print(graph1)

    local min_num = 3
    local conf_thresh = 0.8
    local detections = torch.FloatTensor(100):zero()
    detections[13] = 0.9
    detections[9] = 0.81
    detections[8] = 0.75
    local annotations_plus = torch.FloatTensor(200, 4):zero()
    local ann_total_size = 5

    -- Get initial detections
    local detect_conf, reverse_lookup, active_idx, expanded_idx, edges, edge_conf = graph1:getInitialGraph(detections, annotations_plus, ann_total_size, conf_thresh, min_num)
    print('33333')
    print(detect_conf)
    print(reverse_lookup)
    print(active_idx)
    print(expanded_idx)
    print(edges)
    print(edge_conf)

    -- Now, for each active node, give confidence and expand one
    local num_expand = 3
    local importance = torch.FloatTensor(9):zero()
    importance[1] = 1
    importance[2] = 0.98
    importance[3] = 0.7
    importance[4] = 0.99
    importance[5] = 0.88
    importance[6] = 0.89
    importance[7] = 0.3
    importance[8] = 0.8
    importance[9] = 0.7
    reverse_lookup, active_idx, expanded_idx, edges, edge_conf = graph1:updateGraphFromImportance(importance, reverse_lookup, active_idx, expanded_idx, edges, edge_conf, num_expand)
    print(reverse_lookup)
    print(active_idx)
    print(expanded_idx)
    print(edges)
    print(edge_conf)   


    -- Expand once more
    importance = torch.FloatTensor(12):zero()
    importance[1] = 0.9
    importance[2] = 0.99
    importance[3] = 0.89
    importance[4] = 0.2
    importance[5] = 0.6
    importance[6] = 0.5
    importance[7] = 0.88
    importance[8] = 0.1
    importance[9] = 1
    importance[10] = 0.8
    importance[11] = 0.2
    importance[12] = 0.3
    reverse_lookup, active_idx, expanded_idx, edges, edge_conf = graph1:updateGraphFromImportance(importance, reverse_lookup, active_idx, expanded_idx, edges, edge_conf, num_expand)
    print(reverse_lookup)
    print(active_idx)
    print(expanded_idx)
    print(edges)
    print(edge_conf)    
end

function test_update_graph_from_attention()
    torch.setdefaulttensortype('torch.FloatTensor')

    -- Get graph
    local graph1 = Graph(100, 100)
    graph1:addNewNode("fish", true, true, 42, 13)
    graph1:addNewNode("shark", true, true, 39, 80)
    graph1:addNewNode("aquarium", true, true, 22, 74)
    graph1:addNewNode("blue", true, true, 83, 32)
    graph1:addNewNode("water", true, true, 65, 2)
    graph1:addNewNode("beverage", true, false, 11, -1)
    graph1:addNewNode("milk", true, true, 40, 10)
    graph1:addNewNode("food", true, false, 59, -1)
    graph1:addNewNode("plate", true, true, 38, 24)
    graph1:addNewNode("round_shape", false, true, -1, 9)
    graph1:addNewNode("apple", true, true, 21, 93)
    graph1:addNewNode("orange", true, true, 72, 44)
    graph1:addNewNode("fruit", true, false, 91, -1)
    graph1:addNewNode("green", true, true, 12, 48)
    graph1:addNewNode("grass", true, true, 2, 8)
    graph1:addNewNode("lawn", true, true, 49, 18)
    graph1:addNewNode("house", true, true, 100, 1)
    graph1:addEdge(1, "looks_like", 1, 2, 0.9)
    graph1:addEdge(1, "looks_like", 2, 1, 0.8)
    graph1:addEdge(2, "found_in", 1, 3, 0.87)
    graph1:addEdge(2, "found_in", 2, 3, 0.92)
    graph1:addEdge(3, "has", 3, 4, 0.7)
    graph1:addEdge(4, "part_of", 5, 3, 0.67)
    graph1:addEdge(3, "has", 5, 4, 0.54)
    graph1:addEdge(5, "is", 5, 6, 0.68)
    graph1:addEdge(5, "is", 7, 6, 0.88)
    graph1:addEdge(5, "is", 6, 8, 0.83)
    graph1:addEdge(2, "found_in", 8, 9, 0.99)
    graph1:addEdge(5, "is", 11, 8, 1)
    graph1:addEdge(5, "is", 13, 8, 0.44)
    graph1:addEdge(5, "is", 12, 8, 0.73)
    graph1:addEdge(3, "has", 9, 10, 0.93)
    graph1:addEdge(3, "has", 11, 10, 0.54)
    graph1:addEdge(5, "is", 11, 13, 0.76)
    graph1:addEdge(3, "has", 11, 14, 0.36)
    graph1:addEdge(5, "is", 12, 13, 0.9)
    graph1:addEdge(3, "has", 15, 14, 0.8)
    graph1:addEdge(4, "part_of", 15, 16, 0.7)
    graph1:addEdge(4, "part_of", 17, 16, 0.9)
    print(graph1)

    local min_num = 3
    local conf_thresh = 0.8
    local detections = torch.FloatTensor(100):zero()
    detections[13] = 0.9
    detections[9] = 0.81
    detections[8] = 0.75
    local annotations_plus = torch.FloatTensor(200, 4):zero()
    local ann_total_size = 5

    -- Get initial detections
    local detect_conf, reverse_lookup, active_idx, expanded_idx, edges_interior, edges_frontier = graph1:getInitialGraphWA(detections, annotations_plus, ann_total_size, conf_thresh, min_num)
    print('44444')
    print(detect_conf)
    print(reverse_lookup)
    print(active_idx)
    print(expanded_idx)
    print(edges_interior)
    print(edges_frontier)

    -- Now, for each active node, give confidence and expand one
    local num_expand = 3
    local importance = torch.FloatTensor(9):zero()
    importance[1] = 1
    importance[2] = 0.98
    importance[3] = 0.7
    importance[4] = 0.99
    importance[5] = 0.88
    importance[6] = 0.89
    importance[7] = 0.3
    importance[8] = 0.8
    importance[9] = 0.7
    reverse_lookup, active_idx, expanded_idx, edges_interior, edges_frontier = graph1:updateGraphFromAttention(importance, reverse_lookup, active_idx, expanded_idx, edges_interior, edges_frontier, num_expand)
    print(reverse_lookup)
    print(active_idx)
    print(expanded_idx)
    print(edges_interior)
    print(edges_frontier)   

    -- Expand once more
    importance = torch.FloatTensor(12):zero()
    importance[1] = 0.9
    importance[2] = 0.99
    importance[3] = 0.89
    importance[4] = 0.2
    importance[5] = 0.6
    importance[6] = 0.5
    importance[7] = 0.88
    importance[8] = 0.1
    importance[9] = 1
    importance[10] = 0.8
    importance[11] = 0.2
    importance[12] = 0.3
    reverse_lookup, active_idx, expanded_idx, edges_interior, edges_frontier = graph1:updateGraphFromAttention(importance, reverse_lookup, active_idx, expanded_idx, edges_interior, edges_frontier, num_expand)
    print(reverse_lookup)
    print(active_idx)
    print(expanded_idx)
    print(edges_interior)
    print(edges_frontier)    
end

function test_update_graph_with_lookahead()
    torch.setdefaulttensortype('torch.FloatTensor')

    -- Get graph
    local graph1 = Graph(100, 100)
    graph1:addNewNode("fish", true, true, 42, 13)
    graph1:addNewNode("shark", true, true, 39, 80)
    graph1:addNewNode("aquarium", true, true, 22, 74)
    graph1:addNewNode("blue", true, true, 83, 32)
    graph1:addNewNode("water", true, true, 65, 2)
    graph1:addNewNode("beverage", true, false, 11, -1)
    graph1:addNewNode("milk", true, true, 40, 10)
    graph1:addNewNode("food", true, false, 59, -1)
    graph1:addNewNode("plate", true, true, 38, 24)
    graph1:addNewNode("round_shape", false, true, -1, 9)
    graph1:addNewNode("apple", true, true, 21, 93)
    graph1:addNewNode("orange", true, true, 72, 44)
    graph1:addNewNode("fruit", true, false, 91, -1)
    graph1:addNewNode("green", true, true, 12, 48)
    graph1:addNewNode("grass", true, true, 2, 8)
    graph1:addNewNode("lawn", true, true, 49, 18)
    graph1:addNewNode("house", true, true, 100, 1)
    graph1:addEdge(1, "looks_like", 1, 2, 0.9)
    graph1:addEdge(1, "looks_like", 2, 1, 0.8)
    graph1:addEdge(2, "found_in", 1, 3, 0.87)
    graph1:addEdge(2, "found_in", 2, 3, 0.92)
    graph1:addEdge(3, "has", 3, 4, 0.7)
    graph1:addEdge(4, "part_of", 5, 3, 0.67)
    graph1:addEdge(3, "has", 5, 4, 0.54)
    graph1:addEdge(5, "is", 5, 6, 0.68)
    graph1:addEdge(5, "is", 7, 6, 0.88)
    graph1:addEdge(5, "is", 6, 8, 0.83)
    graph1:addEdge(2, "found_in", 8, 9, 0.99)
    graph1:addEdge(5, "is", 11, 8, 1)
    graph1:addEdge(5, "is", 13, 8, 0.44)
    graph1:addEdge(5, "is", 12, 8, 0.73)
    graph1:addEdge(3, "has", 9, 10, 0.93)
    graph1:addEdge(3, "has", 11, 10, 0.54)
    graph1:addEdge(5, "is", 11, 13, 0.76)
    graph1:addEdge(3, "has", 11, 14, 0.36)
    graph1:addEdge(5, "is", 12, 13, 0.9)
    graph1:addEdge(3, "has", 15, 14, 0.8)
    graph1:addEdge(4, "part_of", 15, 16, 0.7)
    graph1:addEdge(4, "part_of", 17, 16, 0.9)
    print(graph1)

    local min_num = 3
    local conf_thresh = 0.8
    local detections = torch.FloatTensor(100):zero()
    detections[13] = 0.9
    detections[9] = 0.81
    detections[8] = 0.75
    local annotations_plus = torch.FloatTensor(200, 4):zero()
    local ann_total_size = 5

    -- Get initial detections
    local detect_conf, reverse_lookup, active_idx, expanded_idx, edges_interior, edges_frontier = graph1:getInitialNodesFromDetectionsWA(detections, annotations_plus, ann_total_size, conf_thresh, min_num)
    print(detect_conf)
    print(reverse_lookup)
    print(active_idx)
    print(expanded_idx)
    print(edges_interior)
    print(edges_frontier)

    -- Again with another graph
    local graph2 = Graph(200, 200)
    graph2:addNewNode("fern", true, true, 33, 4)
    graph2:addNewNode("green", true, true, 12, 48)
    graph2:addNewNode("apple", true, true, 22, 93)
    graph2:addNewNode("round_shape", false, true, -1, 33)
    graph2:addNewNode("tomato", true, true, 41, 11)
    graph2:addNewNode("clock", true, true, 3, 49)
    graph2:addNewNode("wheel", true, true, 5, 5)
    graph2:addNewNode("tire", true, true, 93, 111)
    graph2:addNewNode("black", true, true, 8, 54)
    graph2:addNewNode("car", true, true, 74, 70)
    graph2:addNewNode("new_york", false, true, -1, 23)
    graph2:addNewNode("statue_of_liberty", false, true, -1, 80)
    graph2:addNewNode("raceway", true, true, 40, 99)
    graph2:addNewNode("taxi", true, true, 65, 20)
    graph2:addNewNode("car_dealership", false, true, -1, 55)
    graph2:addNewNode("street", true, true, 35, 81)
    graph2:addNewNode("snowy_weather", false, true, -1, 100)
    graph2:addEdge(1, "has", 1, 2, 0.9)
    graph2:addEdge(1, "has", 3, 2, 0.8)
    graph2:addEdge(2, "looks_like", 5, 3, 0.7)
    graph2:addEdge(2, "looks_like", 3, 5, 0.5)
    graph2:addEdge(1, "has", 3, 4, 1)
    graph2:addEdge(1, "has", 6, 4 ,0.89)
    graph2:addEdge(1, "has", 7, 4, 0.8)
    graph2:addEdge(1, "has", 8, 4, 0.77)
    graph2:addEdge(2, "looks_like", 7, 8, 0.8)
    graph2:addEdge(2, "looks_like", 8, 7, 0.98)
    graph2:addEdge(1, "has", 8, 9, 1)
    graph2:addEdge(3, "part_of", 7, 10, .9)
    graph2:addEdge(4, "found_in", 10, 11, 0.88)
    graph2:addEdge(1, "has", 11, 12, 0.58)
    graph2:addEdge(4, "found_in", 10, 13, 0.78)
    graph2:addEdge(4, "found_in", 14, 13, 0.86)
    graph2:addEdge(4, "found_in", 10, 15, 0.78)
    graph2:addEdge(4, "found_in", 10, 16, 0.78)
    graph2:addEdge(1, "has", 16, 17, 0.88)
    print(graph2)

    local min_num = 3
    local conf_thresh = 0.5
    local detections = torch.FloatTensor(200):zero()
    detections[93] = 0.9
    detections[11] = 0.81
    detections[55] = 0.75
    detections[48] = 0.99
    local annotations_plus = torch.FloatTensor(200, 4):zero()
    local ann_total_size = 5

    -- Get initial detections
    local detect_conf, reverse_lookup, active_idx, expanded_idx, edges_interior, edges_frontier = graph2:getInitialNodesFromDetectionsWA(detections, annotations_plus, ann_total_size, conf_thresh, min_num)
    print(detect_conf)
    print(reverse_lookup)
    print(active_idx)
    print(expanded_idx)
    print(edges_interior)
    print(edges_frontier)

    -- Redo graph1 expansion
    local min_num = 3
    local conf_thresh = 0.8
    local detections = torch.FloatTensor(100):zero()
    detections[13] = 0.9
    detections[9] = 0.81
    detections[8] = 0.75
    local annotations_plus = torch.FloatTensor(200, 4):zero()
    local ann_total_size = 5
    local detect_conf, reverse_lookup, active_idx, expanded_idx, edges_interior, edges_frontier = graph1:getInitialNodesFromDetectionsWA(detections, annotations_plus, ann_total_size, conf_thresh, min_num)
   
    -- test updateGraphWithLookahead by expanding with some values
    local num_expand = 3
    local scores = torch.Tensor(7):zero()
    scores[1] = 1
    scores[2] = 0.99
    scores[4] = 0.8
    scores[6] = 0.5 
 
    reverse_lookup, active_idx, expanded_idx, edges_interior, edges_frontier = graph1:updateGraphWithLookahead(scores, reverse_lookup, active_idx, expanded_idx, edges_interior, edges_frontier, num_expand)
    print(reverse_lookup) 
    print(active_idx)
    print(expanded_idx)
    print(edges_interior)
    print(edges_frontier) 

    -- test again with one more expansion
    scores = torch.Tensor(6):zero()
    scores[1] = 0.5
    scores[5] = 1
    scores[6] = 0.98
    scores[4] = 0.75
    
    reverse_lookup, active_idx, expanded_idx, edges_interior, edges_frontier = graph1:updateGraphWithLookahead(scores, reverse_lookup, active_idx, expanded_idx, edges_interior, edges_frontier, num_expand)
    print(reverse_lookup) 
    print(active_idx)
    print(expanded_idx)
    print(edges_interior)
    print(edges_frontier)
 
    -- Now, for each active node, give confidence and expand one
    --[[local num_expand = 3
    local importance = torch.FloatTensor(9):zero()
    importance[1] = 1
    importance[2] = 0.98
    importance[3] = 0.7
    importance[4] = 0.99
    importance[5] = 0.88
    importance[6] = 0.89
    importance[7] = 0.3
    importance[8] = 0.8
    importance[9] = 0.7
    reverse_lookup, active_idx, expanded_idx, edges_interior, edges_frontier = graph1:updateGraphFromAttention(importance, reverse_lookup, active_idx, expanded_idx, edges_interior, edges_frontier, num_expand)
    print(reverse_lookup)
    print(active_idx)
    print(expanded_idx)
    print(edges_interior)
    print(edges_frontier)   

    -- Expand once more
    importance = torch.FloatTensor(12):zero()
    importance[1] = 0.9
    importance[2] = 0.99
    importance[3] = 0.89
    importance[4] = 0.2
    importance[5] = 0.6
    importance[6] = 0.5
    importance[7] = 0.88
    importance[8] = 0.1
    importance[9] = 1
    importance[10] = 0.8
    importance[11] = 0.2
    importance[12] = 0.3
    reverse_lookup, active_idx, expanded_idx, edges_interior, edges_frontier = graph1:updateGraphFromAttention(importance, reverse_lookup, active_idx, expanded_idx, edges_interior, edges_frontier, num_expand)
    print(reverse_lookup)
    print(active_idx)
    print(expanded_idx)
    print(edges_interior)
    print(edges_frontier)    ]]--
end

function test_get_discounted_values()
    -- Start with two graphs we've already made
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
    print(graph1)
    local graph2 = Graph(100, 100)
    graph2:addNewNode("fish", true, true, 42, 13)
    graph2:addNewNode("shark", true, true, 39, 80)
    graph2:addNewNode("aquarium", true, true, 22, 74)
    graph2:addNewNode("blue", true, true, 83, 32)
    graph2:addNewNode("water", true, true, 65, 2)
    graph2:addNewNode("beverage", true, false, 11, -1)
    graph2:addNewNode("milk", true, true, 40, 10)
    graph2:addNewNode("food", true, false, 59, -1)
    graph2:addNewNode("plate", true, true, 38, 24)
    graph2:addNewNode("round_shape", false, true, -1, 9)
    graph2:addNewNode("apple", true, true, 21, 93)
    graph2:addNewNode("orange", true, true, 72, 44)
    graph2:addNewNode("fruit", true, false, 91, -1)
    graph2:addNewNode("green", true, true, 12, 48)
    graph2:addNewNode("grass", true, true, 2, 8)
    graph2:addNewNode("lawn", true, true, 49, 18)
    graph2:addNewNode("house", true, true, 100, 1)
    graph2:addEdge(1, "looks_like", 1, 2, 0.9)
    graph2:addEdge(1, "looks_like", 2, 1, 0.8)
    graph2:addEdge(2, "found_in", 1, 3, 0.87)
    graph2:addEdge(2, "found_in", 2, 3, 0.92)
    graph2:addEdge(3, "has", 3, 4, 0.7)
    graph2:addEdge(4, "part_of", 5, 3, 0.67)
    graph2:addEdge(3, "has", 5, 4, 0.54)
    graph2:addEdge(5, "is", 5, 6, 0.68)
    graph2:addEdge(5, "is", 7, 6, 0.88)
    graph2:addEdge(5, "is", 6, 8, 0.83)
    graph2:addEdge(2, "found_in", 8, 9, 0.99)
    graph2:addEdge(5, "is", 11, 8, 1)
    graph2:addEdge(5, "is", 13, 8, 0.44)
    graph2:addEdge(5, "is", 12, 8, 0.73)
    graph2:addEdge(3, "has", 9, 10, 0.93)
    graph2:addEdge(3, "has", 11, 10, 0.54)
    graph2:addEdge(5, "is", 11, 13, 0.76)
    graph2:addEdge(3, "has", 11, 14, 0.36)
    graph2:addEdge(5, "is", 12, 13, 0.9)
    graph2:addEdge(3, "has", 15, 14, 0.8)
    graph2:addEdge(4, "part_of", 15, 16, 0.7)
    graph2:addEdge(4, "part_of", 17, 16, 0.9)
    print(graph2)

    -- Test a couple of discounted value calculations
    local vocab_target_idx = {65, 74, 3, 1} 
    local gamma = 0.9
    local num_steps = 3
    local node_values = graph1:getDiscountedValues(vocab_target_idx, gamma, num_steps)
    print(node_values)

    -- Another test
    vocab_target_idx = {35, 12, 100}
    num_steps = 5
    node_values = graph1:getDiscountedValues(vocab_target_idx, gamma, num_steps)
    print(node_values)

    -- Another
    vocab_target_idx = {42, 39, 12, 100, 44, 45}
    num_steps = 5
    node_values = graph2:getDiscountedValues(vocab_target_idx, gamma, num_steps)
    print(node_values)
end

function test_get_edges()
    -- Make graph
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
    print(graph1)

    -- Make sure getFullGraph works
    edges = graph1:getFullGraph()
    print(edges)
end

function test_check_edge_exists()
    -- Make graph
    local graph1 = Graph(100, 100)
    graph1:addNewNode("fish", true, true, 42, 13)
    graph1:addNewNode("shark", true, true, 39, 80)
    graph1:addNewNode("aquarium", true, true, 22, 74)
    graph1:addNewNode("blue", true, true, 83, 32)
    graph1:addNewNode("water", true, true, 65, 2)
    graph1:addNewNode("beverage", true, false, 11, -1)
    graph1:addNewNode("milk", true, true, 40, 10)
    graph1:addNewNode("food", true, false, 59, -1)
    graph1:addNewNode("plate", true, true, 38, 24)
    graph1:addNewNode("round_shape", false, true, -1, 9)
    graph1:addNewNode("apple", true, true, 21, 93)
    graph1:addNewNode("orange", true, true, 72, 44)
    graph1:addNewNode("fruit", true, false, 91, -1)
    graph1:addNewNode("green", true, true, 12, 48)
    graph1:addNewNode("grass", true, true, 2, 8)
    graph1:addNewNode("lawn", true, true, 49, 18)
    graph1:addNewNode("house", true, true, 100, 1)
    graph1:addEdge(1, "looks_like", 1, 2, 0.9)
    graph1:addEdge(1, "looks_like", 2, 1, 0.8)
    graph1:addEdge(2, "found_in", 1, 3, 0.87)
    graph1:addEdge(2, "found_in", 2, 3, 0.92)
    graph1:addEdge(3, "has", 3, 4, 0.7)
    graph1:addEdge(4, "part_of", 5, 3, 0.67)
    graph1:addEdge(3, "has", 5, 4, 0.54)
    graph1:addEdge(5, "is", 5, 6, 0.68)
    graph1:addEdge(5, "is", 7, 6, 0.88)
    graph1:addEdge(5, "is", 6, 8, 0.83)
    graph1:addEdge(2, "found_in", 8, 9, 0.99)
    graph1:addEdge(5, "is", 11, 8, 1)
    graph1:addEdge(5, "is", 13, 8, 0.44)
    graph1:addEdge(5, "is", 12, 8, 0.73)
    graph1:addEdge(3, "has", 9, 10, 0.93)
    graph1:addEdge(3, "has", 11, 10, 0.54)
    graph1:addEdge(5, "is", 11, 13, 0.76)
    graph1:addEdge(3, "has", 11, 14, 0.36)
    graph1:addEdge(5, "is", 12, 13, 0.9)
    graph1:addEdge(3, "has", 15, 14, 0.8)
    graph1:addEdge(4, "part_of", 15, 16, 0.7)
    graph1:addEdge(4, "part_of", 17, 16, 0.9)
   
    local edge_exists, edge_idx = graph1:checkEdgeExists(5, "is", 5, 6)
    print(edge_exists)
    print(edge_idx)
    edge_exists, edge_idx = graph1:checkEdgeExists(2, "found_in", 8, 9)
    print(edge_exists)
    print(edge_idx)
    edge_exists, edge_idx = graph1:checkEdgeExists(4, "part_of", 15, 16)
    print(edge_exists)
    print(edge_idx)
    edge_exists, edge_idx = graph1:checkEdgeExists(1, "looks_like", 15, 16)
    print(edge_exists)
    print(edge_idx)
    edge_exists, edge_idx = graph1:checkEdgeExists(3, "has", 4, 3)
    print(edge_exists)
    print(edge_idx)
end

-------------------- run tests -------------------------

time_func(test_node_create)
time_func(test_edge_create)
time_func(test_graph_create)
time_func(test_getexpanded)
time_func(test_getexpandedwa)
time_func(test_get_initial_graph)
time_func(test_get_initial_graph_wa)
time_func(test_update_graph_from_importance)
time_func(test_update_graph_from_attention)
time_func(test_get_discounted_values)
time_func(test_get_edges)
time_func(test_check_edge_exists)
time_func(test_update_graph_with_lookahead)
