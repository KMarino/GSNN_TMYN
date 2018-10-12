-- Kenneth Marino
-- 06/2016
-- Class to deal with graph-based operations efficiently and easily

require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')
local ffi = require 'ffi'
local class = require('pl.class')
local dir = require 'pl.dir'
local tablex = require 'pl.tablex'
local argcheck = require 'argcheck'
require 'sys'
require 'xlua'
require 'json'

-- Node class
local Node = torch.class('Node')

local nodecheck = argcheck{
    {name="index", type="number"},
    {name="name", type="string"},
    {name="in_vocab", type="boolean"},
    {name="has_neil_detector", type="boolean"},
    {name="vocab_index", type="number"},
    {name="neil_detector_index", type="number"},
}

function Node:__init(...)
    local index, name, in_vocab, has_neil_detector, vocab_index, neil_detector_index = nodecheck(...)
        
    -- Index in graph
    self.index = index

    -- Name of node
    self.name = name

    -- Whether it is in vocabulary of captioning task (or other classification task)
    self.in_vocab = in_vocab

    -- Whether this node has an associated NEIL detector
    self.has_neil_detector = has_neil_detector

    -- Index into caption vocabulary. -1 if it's not in vocab
    assert((not in_vocab and vocab_index == -1) or (in_vocab and vocab_index > 0))
    self.vocab_index = vocab_index

    -- Index into neil detectors. -1 if it's not a detector
    assert((not has_neil_detector and neil_detector_index == -1) or (has_neil_detector and neil_detector_index > 0))
    self.neil_detector_index = neil_detector_index

    -- Set up edge lists
    self.outgoing_edges = {}
    self.incoming_edges = {}
end

function Node:__tostring__()
    local str = "Node: " .. self.index .. ", " .. self.name .. ", in_vocab " .. tostring(self.in_vocab) .. ", has_neil " .. tostring(self.has_neil_detector) .. ", vocab_index " .. self.vocab_index .. ", neil_ind " .. tostring(self.neil_detector_index)
    return str
end

-- Edgetype class
local EdgeType = torch.class('EdgeType')

local edgetypecheck = argcheck{
    {name="index", type="number"},
    {name="name", type="string"},
}

function EdgeType:__init(...)
    local index, name = edgetypecheck(...)
    self.index = index
    self.name = name
end

function EdgeType:__tostring__()
    local str = "EdgeType: " .. self.index .. ", " .. self.name
    return str
end

-- Edge class
local Edge = torch.class('Edge')

local edgecheck = argcheck{
    {name="edgetype", type="EdgeType"},
    {name="start_node", type="Node"},
    {name="end_node", type="Node"},
    {name="index", type="number"},
    {name="confidence", type="number", default=1}
}

function Edge:__init(...)
    local edgetype, start_node, end_node, index, confidence = edgecheck(...)
    self.edgetype = edgetype
    self.start_node = start_node
    self.end_node = end_node
    self.confidence = confidence
    self.index = index
end

function Edge:__tostring__()
    local str = "Edge: " .. self.index .. ", " .. self.start_node.name .. ", " .. self.end_node.name .. ", " .. self.confidence .. ", " .. tostring(self.edgetype)
    return str
end

-- Graph class
local Graph = torch.class('Graph')

local graphcheck = argcheck{
    {name="detector_size", type="number", default=0},
    {name="vocab_size", type="number", default=0},
    {name="filename", type="string", default=""},
}

-- Constructor
function Graph:__init(...)
    local detector_size, vocab_size, filename = graphcheck(...)
   
    self.detector_size = detector_size
    self.vocab_size = vocab_size
    self.detector_reverse_lookup = torch.LongTensor(self.detector_size):fill(-1)
    self.vocab_reverse_lookup = torch.LongTensor(self.vocab_size):fill(-1)

    -- Default is empty graph
    self.n_total_nodes = 0
    self.n_total_edges = 0  
    self.n_edge_types = 0
    self.nodes = {}                -- Table of nodes in the graph
    self.edges = {}
    self.neil_detector_nodes = {}  -- Table of nodes that have NEIL detections
    self.vocab_nodes = {}          -- Table of nodes that are in vocab
    self.edge_types = {}
    if filename ~= '' then
        local graph_data = torch.load(filename)
  
        -- Load basic info
        self.detector_size = graph_data.detector_size
        self.vocab_size = graph_data.vocab_size 
        self.detector_reverse_lookup = torch.LongTensor(self.detector_size):fill(-1)
        self.vocab_reverse_lookup = torch.LongTensor(self.vocab_size):fill(-1)
              
        -- Add nodes
        for i = 1, #graph_data.nodelist do
            local node_data = graph_data.nodelist[i]
            self:addNewNode(node_data.name, node_data.in_vocab, node_data.has_neil_detector, node_data.vocab_index, node_data.neil_detector_index)            
        end

        -- Add edge types
        for i = 1, #graph_data.edge_type_list do
            local edge_type_data = graph_data.edge_type_list[i]
            self:addEdgeType(i, edge_type_data.name)
        end       
 
        -- Add edges
        for i = 1, #graph_data.edge_list do
            local edge_data = graph_data.edge_list[i]
            local edge_type_ind = edge_data.edgetype_ind
            local edge_type_name = self.edge_types[edge_type_ind].name
            self:addEdge(edge_type_ind, edge_type_name, edge_data.start_node, edge_data.end_node, edge_data.confidence)
        end 

        -- Check
        assert(self.n_total_nodes == graph_data.n_total_nodes)
        assert(self.n_total_edges == graph_data.n_total_edges)
        assert(self.n_edge_types == graph_data.n_edge_types)
    end
end

-- Function to save graph eficiently
function Graph:save(filename)
    local graph_data = {}
    
    -- Save basic info
    graph_data.detector_size = self.detector_size
    graph_data.vocab_size = self.vocab_size
    graph_data.n_total_nodes = self.n_total_nodes
    graph_data.n_total_edges = self.n_total_edges
    graph_data.n_edge_types = self.n_edge_types

    -- Save graph node data
    graph_data.nodelist = {}
    for i = 1, #self.nodes do
        local node = self.nodes[i]
        local node_data = {}
        node_data.name = node.name
        node_data.in_vocab = node.in_vocab
        node_data.vocab_index = node.vocab_index
        node_data.has_neil_detector = node.has_neil_detector
        node_data.neil_detector_index = node.neil_detector_index    
        graph_data.nodelist[i] = node_data
    end
    assert(#graph_data.nodelist == graph_data.n_total_nodes)

    -- Save edge types
    graph_data.edge_type_list = {}
    for i = 1, #self.edge_types do
        local edge_type = self.edge_types[i]
        local edge_type_data = {}
        edge_type_data.name = edge_type.name
        graph_data.edge_type_list[i] = edge_type_data
    end
    assert(#graph_data.edge_type_list == graph_data.n_edge_types)

    -- Save edges 
    graph_data.edge_list = {}
    for i = 1, #self.edges do 
        local edge = self.edges[i]
        local edge_data = {}
        edge_data.edgetype_ind = edge.edgetype.index
        edge_data.start_node = edge.start_node.index
        edge_data.end_node = edge.end_node.index
        edge_data.confidence = edge.confidence
        graph_data.edge_list[i] = edge_data    
    end 
    assert(#graph_data.edge_list == graph_data.n_total_edges)

    -- Save to file
    torch.save(filename, graph_data)
end

function Graph:__tostring__()
    local str = "Graph: #nodes " .. self.n_total_nodes .. ", #edges " .. self.n_total_edges .. ", #edgetypes " .. self.n_edge_types .. "\n"
    str = str .. "  Nodes:\n"
    for i = 1, self.n_total_nodes do
        str = str .. "    " .. tostring(self.nodes[i]) .. "\n"
    end
    str = str .. "  Edges:\n"
    for i = 1, self.n_total_edges do
        str = str .. "    " .. tostring(self.edges[i]) .. "\n"
    end 
    str = str .. "  EdgeTypes:\n"
    for i = 1, self.n_edge_types do
        str = str .. "    " .. tostring(self.edge_types[i]) .. "\n"
    end
    str = str .. "  Neil Detector Nodes: {"
    for i = 1, #self.neil_detector_nodes do
        str = str .. self.neil_detector_nodes[i].index .. " "
    end
    str = str .. "}\n"
    str = str .. "  Vocab Nodes: {"
    for i = 1, #self.vocab_nodes do
        str = str .. self.vocab_nodes[i].index .. " "
    end
    str = str .. "}"
    return str
end

function Graph:saveAsJSON(filename)
    local jsondata = {}
    jsondata.nodes = {}
    for i = 1, self.n_total_nodes do
        table.insert(jsondata.nodes, self.nodes[i].name)
    end
    jsondata.edge_types = {}
    for i = 1, self.n_edge_types do
        table.insert(jsondata.edge_types, self.edge_types[i].name)
    end
    jsondata.edges = {}
    for i = 1, self.n_total_edges do
        local edge = {self.edges[i].start_node.index, self.edges[i].edgetype.index, self.edges[i].end_node.index}
        table.insert(jsondata.edges, edge)
    end
    json.save(filename, jsondata)
end

-- Add empty node to graph
function Graph:addNewNode(name, in_vocab, has_neil_detector, vocab_index, neil_detector_index)
    -- Create node
    local newnode = Node.new(self.n_total_nodes+1, name, in_vocab, has_neil_detector, vocab_index, neil_detector_index)

    -- Update counts and lists 
    self.n_total_nodes = self.n_total_nodes + 1
    table.insert(self.nodes, newnode)
    if newnode.in_vocab then
        table.insert(self.vocab_nodes, newnode)
        self.vocab_reverse_lookup[vocab_index] = self.n_total_nodes
    end
    if newnode.has_neil_detector then
        table.insert(self.neil_detector_nodes, newnode)
        self.detector_reverse_lookup[neil_detector_index] = self.n_total_nodes
    end
end

-- Add edge types without adding edges
function Graph:addEdgeType(edge_type_idx, edge_type_name)
    assert(edge_type_idx == self.n_edge_types + 1)
    local edgetype = EdgeType.new(edge_type_idx, edge_type_name)
    table.insert(self.edge_types, edgetype)
    self.n_edge_types = self.n_edge_types + 1 
end

-- Add new edge to graph
function Graph:addEdge(edge_type_idx, edge_type_name, start_node_idx, end_node_idx, confidence)
    -- Get (or create) edge type
    local edgetype
    if edge_type_idx > self.n_edge_types then
        assert(edge_type_idx == self.n_edge_types + 1)
        edgetype = EdgeType.new(edge_type_idx, edge_type_name)
        table.insert(self.edge_types, edgetype)
        self.n_edge_types = self.n_edge_types + 1 
    else
        edgetype = self.edge_types[edge_type_idx]
        assert(edge_type_name == edgetype.name)
    end

    -- Create edge
    local startnode = self.nodes[start_node_idx]
    local endnode = self.nodes[end_node_idx]
    local edge = Edge.new(edgetype, startnode, endnode, self.n_total_edges+1, confidence)
    table.insert(startnode.outgoing_edges, edge)
    table.insert(endnode.incoming_edges, edge)
    table.insert(self.edges, edge)
    self.n_total_edges = self.n_total_edges + 1
end

-- Check if edge alreadt exists
function Graph:checkEdgeExists(edge_type_idx, edge_type_name, start_node_idx, end_node_idx) 
    local edge_exists = false
    local edge_idx = -1
    
    -- Get start node
    local startnode = self.nodes[start_node_idx]
    local outgoing = startnode.outgoing_edges
    for i = 1, #outgoing do
        local edge = outgoing[i]
        if edge.edgetype.index == edge_type_idx and edge.start_node.index == start_node_idx and edge.end_node.index == end_node_idx then      
            assert(edge_type_name == edge.edgetype.name)
            edge_exists = true
            edge_idx = edge.index
            break
        end
    end

    return edge_exists, edge_idx
end

-- Get the edges of the entire graph
function Graph:getFullGraph()
    local edges = {}
    for i, edge in ipairs(self.edges) do
        local e = {edge.start_node.index, edge.edgetype.index, edge.end_node.index}
        table.insert(edges, e)
    end
    return {edges}
end

-- Get list of active nodes from detections
function Graph:getInitialNodesFromDetections(init_det_orig, annotations_orig, ann_total_size, conf_thresh, min_num)
    --print(torch.type(annotations_orig))

    -- Make copy of initial_detections
    local initial_detections = torch.Tensor(init_det_orig:size()):copy(init_det_orig)
    
    -- Get detector indices to select
    local above_thresh = torch.gt(initial_detections, conf_thresh)
    local detect_inds = {}
    local annotations = {}
    if above_thresh:sum() < min_num then
        for i = 1, min_num do
            local m, j = torch.max(initial_detections, 1)
            while torch.type(j) == 'torch.LongTensor' do
                m = m[1]
                j = j[1]
            end

            table.insert(detect_inds, j)            
            local ann = {}
            table.insert(ann, m)
            for k = 1, ann_total_size-1 do
                --print(annotations_orig)
                --print(j)
                --print(k)
                --print(ann)
                --print(annotations_orig[j])
                table.insert(ann, annotations_orig[j][k])
            end
            table.insert(annotations, ann)
            initial_detections[j] = -1
        end
    else
        for i = 1, self.detector_size do
            --print(conf_thresh)
            --print(torch.type(conf_thresh))
            --print(torch.type(initial_detections))
            --print(initial_detections:size())
            --print(torch.type(initial_detections[i]))
            --print(initial_detections[i])
            if initial_detections[i] > conf_thresh then
                table.insert(detect_inds, i)
                local ann = {}
                table.insert(ann, initial_detections[i])
                for k = 1, ann_total_size-1 do
                    table.insert(ann, annotations_orig[i][k])
                end
                table.insert(annotations, ann)
            end
        end
    end

    -- Get indices in graph
    local active_idx = {}
    local reverse_lookup = torch.LongTensor(self.n_total_nodes):fill(-1)
    for i = 1, #detect_inds do
        local detect_ind = detect_inds[i]
        local graph_ind = self.detector_reverse_lookup[detect_ind]
        assert(graph_ind ~= -1)
        table.insert(active_idx, graph_ind)
        reverse_lookup[graph_ind] = i
    end

    return active_idx, reverse_lookup, annotations 
end

-- Given the initial detections, give the graph with the initially detected classes and their neighbors
function Graph:getInitialGraph(init_det_orig, annotations_orig, ann_total_size, conf_thresh, min_num)
    -- Get the initial nodes
    local active_idx, reverse_lookup, annotations = Graph.getInitialNodesFromDetections(self, init_det_orig, annotations_orig, ann_total_size, conf_thresh, min_num)

    local expand_idx = {}
    for i = 1, #active_idx do
        table.insert(expand_idx, active_idx[i])
    end   

    -- Get initial edges between all the activated nodes
    local edges = {}
    local edge_conf = {}
  
    -- Go through edges to new nodes and add them (if they are in our subgraph)
    for i = 1, #active_idx do
        local cur_node_idx = active_idx[i]
        local cur_node = self.nodes[cur_node_idx]
        for j = 1, #cur_node.outgoing_edges do
   
            local cur_edge = cur_node.outgoing_edges[j]
            local other_node_idx = cur_edge.end_node.index

            -- If other end of edge is in graph, add edge
            if reverse_lookup[other_node_idx] ~= -1 then
                local newedge = {reverse_lookup[cur_node_idx], cur_edge.edgetype.index, reverse_lookup[other_node_idx]}
                table.insert(edges, newedge)
                table.insert(edge_conf, cur_edge.confidence)
            end
        end
    end       

    -- Now call graph expansion 
    local reverse_lookup, active_idx, edges, edge_conf = self:getExpandedGraph(reverse_lookup, active_idx, expand_idx, edges, edge_conf)
   
    return annotations, reverse_lookup, active_idx, expand_idx, edges, edge_conf
end


-- Given the initial detections, give the graph with the initially detected classes only. Expansion should then be handled by GSNN's attention
function Graph:getInitialNodesFromDetectionsWA(init_det_orig, annotations_orig, ann_total_size, conf_thresh, min_num)
    -- Get the initial nodes as before
    local active_idx, reverse_lookup, annotations = Graph.getInitialNodesFromDetections(self, init_det_orig, annotations_orig, ann_total_size, conf_thresh, min_num)
    local expand_idx = {}

    -- Get initial edges between all the activated nodes
    local edges_interior = {}
    local edges_frontier = {}
  
    -- Go through edges to new nodes and add them (if they are in our subgraph)
    for i = 1, #active_idx do
        --print('Adding node: ' .. active_idx[i])
        local cur_node_idx = active_idx[i]
        local cur_node = self.nodes[cur_node_idx]
        
        for j = 1, #cur_node.outgoing_edges do
            local cur_edge = cur_node.outgoing_edges[j]
            local other_node_idx = cur_edge.end_node.index
            local edge_struct = {}
            --print('Adding edge: ' .. cur_edge.index)
            edge_struct.id = cur_edge.index
            edge_struct.edge = {reverse_lookup[cur_node_idx], cur_edge.edgetype.index, reverse_lookup[other_node_idx]}

            -- If other end of edge is in graph, add edge
            if reverse_lookup[other_node_idx] ~= -1 then
                table.insert(edges_interior, edge_struct)
            else
                edges_frontier[cur_edge.index] = edge_struct
            end
        end
        for j = 1, #cur_node.incoming_edges do
            local cur_edge = cur_node.incoming_edges[j]
            local other_node_idx = cur_edge.start_node.index
            local edge_struct = {}
            --print('Adding edge: ' .. cur_edge.index)
            edge_struct.id = cur_edge.index
            edge_struct.edge = {reverse_lookup[other_node_idx], cur_edge.edgetype.index, reverse_lookup[cur_node_idx]}
 
            -- If other end is not in graph, add as frontier edge
            if reverse_lookup[other_node_idx] == -1 then
                edges_frontier[cur_edge.index] = edge_struct
            end
        end
    end       

    return annotations, reverse_lookup, active_idx, expand_idx, edges_interior, edges_frontier
end

-- Given the initial detections, give the graph with the initially detected classes and their neighbors
-- This version returns interior edge list and exterior edge list
function Graph:getInitialGraphWA(init_det_orig, annotations_orig, ann_total_size, conf_thresh, min_num)
    -- Get the initial nodes
    local active_idx, reverse_lookup, annotations = Graph.getInitialNodesFromDetections(self, init_det_orig, annotations_orig, ann_total_size, conf_thresh, min_num)

    local expand_idx = {}
    for i = 1, #active_idx do
        table.insert(expand_idx, active_idx[i])
    end   

    -- Get initial edges between all the activated nodes
    local edges_interior = {}
    local edges_frontier = {}
  
    -- Go through edges to new nodes and add them (if they are in our subgraph)
    for i = 1, #active_idx do
        local cur_node_idx = active_idx[i]
        local cur_node = self.nodes[cur_node_idx]
        
        for j = 1, #cur_node.outgoing_edges do
            local cur_edge = cur_node.outgoing_edges[j]
            local other_node_idx = cur_edge.end_node.index
            local edge_struct = {}
            edge_struct.id = cur_edge.index
            edge_struct.edge = {reverse_lookup[cur_node_idx], cur_edge.edgetype.index, reverse_lookup[other_node_idx]}

            -- If other end of edge is in graph, add edge
            if reverse_lookup[other_node_idx] ~= -1 then
                table.insert(edges_interior, edge_struct)
            else
                edges_frontier[cur_edge.index] = edge_struct
            end
        end
        for j = 1, #cur_node.incoming_edges do
            local cur_edge = cur_node.incoming_edges[j]
            local other_node_idx = cur_edge.start_node.index
            local edge_struct = {}
            edge_struct.id = cur_edge.index
            edge_struct.edge = {reverse_lookup[other_node_idx], cur_edge.edgetype.index, reverse_lookup[cur_node_idx]}
 
            -- If other end is not in graph, add as frontier edge
            if reverse_lookup[other_node_idx] == -1 then
                edges_frontier[cur_edge.index] = edge_struct
            end
        end
    end       

    -- Now call graph expansion 
    local reverse_lookup, active_idx, edges_interior, edges_frontier = self:getExpandedGraphWA(reverse_lookup, active_idx, expand_idx, edges_interior, edges_frontier)
   
    return annotations, reverse_lookup, active_idx, expand_idx, edges_interior, edges_frontier
end

-- Given forintier edge values from edge net, update the graph
function Graph:updateGraphWithLookahead(scores_orig, reverse_lookup, active_idx, expanded_idx, edges_interior, edges_frontier, num_expand, trainmode)
    -- Check cases where all nodes expanded or no frontier edges (equivalent?)
    local num_edges = 0
    for _ in pairs(edges_frontier) do num_edges = num_edges + 1 end
    if num_edges == 0 then
        -- Should have no more nodes to expand, scores should be nil and no more edges on frontier
        assert(num_edges == 0 and not scores_orig)
        -- Nohting to be done, return
        return reverse_lookup, active_idx, expanded_idx, edges_interior, edges_frontier
    end
    --print(scores_orig)
    assert(scores_orig:size(1) == num_edges)

    -- Make copy of edge scores
    local edge_scores = torch.Tensor(scores_orig:size()):copy(scores_orig)
    
    -- Get a table of frontier edges and reverse lookup into score_orig
    local frontier_edge_to_score_dict = {}
    local frontier_edge_table = {}
    local count = 1
    for _, edge in pairs(edges_frontier) do
        table.insert(frontier_edge_table, edge)
        frontier_edge_to_score_dict[edge.id] = count
        count = count + 1
    end
    assert(#frontier_edge_table == num_edges)

    -- Go through edges scores until we've added num_expand new nodes, or run out of edges
    local num_edges_left = num_edges
    local num_added = 0
    --print(trainmode)
    while num_added < num_expand and num_edges_left > 0 do
        -- If in train mode, with prob epsilon (hardcoded 0.05 right now) choose randomly
        local m, j
        if torch.uniform() < 0.05 and trainmode then
            j = -1
            m = -1
            while m == -1 do 
                j = (torch.random() % edge_scores:size(1)) + 1
                assert(j > 0)
                assert(j <= edge_scores:size(1))
                m = edge_scores[j]
                while not (torch.type(m) == 'number') do
                    m = m[1]
                end
            end
        -- Otherwise greedy
        else
            -- Get max edge
            m, j = torch.max(edge_scores, 1)
            while not (torch.type(j) == 'number') do
                j = j[1]
                m = m[1]
            end
        end
        assert(torch.type(j) == 'number')
        assert(torch.type(m) == 'number')
        assert(m ~= -1)
        local best_edge_ind = frontier_edge_table[j].id
        local best_edge = self.edges[best_edge_ind]
        --print(best_edge) 
        local src_idx = best_edge.start_node.index
        local dst_idx = best_edge.end_node.index

        -- It should always expands new node, otherwise we're not updating right
        -- Really this should never happen, but it's some weird edge case, so
        if (reverse_lookup[src_idx] ~= -1 and reverse_lookup[dst_idx] ~= -1) then
            -- This really should never happen
            edge_scores[j] = -1
            num_edges_left = num_edges_left - 1
        
        -- Secret else block here    
        else

        -- Get the new node
        local new_node
        if reverse_lookup[src_idx] == -1 then
            assert(reverse_lookup[dst_idx] ~= -1)
            new_node = best_edge.start_node
        elseif reverse_lookup[dst_idx] == -1 then
            assert(reverse_lookup[src_idx] ~= -1)
            new_node = best_edge.end_node
        else
            error('You shouldnt get here')
        end

        -- Add to active, and update added count and reverse_lookup
        table.insert(active_idx, new_node.index)
        reverse_lookup[new_node.index] = #active_idx
        num_added = num_added + 1
       
        for i = 1, #new_node.outgoing_edges + #new_node.incoming_edges do
            local cur_edge
            local edge_struct = {}
            local other_node
            if i <= #new_node.outgoing_edges then
                cur_edge = new_node.outgoing_edges[i]
                other_node = cur_edge.end_node
                edge_struct.edge = {reverse_lookup[new_node.index], cur_edge.edgetype.index, reverse_lookup[other_node.index]}
                edge_struct.id = cur_edge.index
            else
                cur_edge = new_node.incoming_edges[i - #new_node.outgoing_edges]
                other_node = cur_edge.start_node
                edge_struct.edge = {reverse_lookup[other_node.index], cur_edge.edgetype.index, reverse_lookup[new_node.index]}
                edge_struct.id = cur_edge.index
            end       

            -- If other node is not in graph, add the edge to edges_frontier
            if reverse_lookup[other_node.index] == -1 then
                edges_frontier[cur_edge.index] = edge_struct
            -- If conneccts to another active_index, remove from frontier, add to interior, update edge_scores and num_edges_left
            else
                edges_frontier[cur_edge.index] = nil
                table.insert(edges_interior, edge_struct)
                
                local score_idx = frontier_edge_to_score_dict[cur_edge.index]
                if score_idx then
                    edge_scores[score_idx] = -1
                    num_edges_left = num_edges_left - 1
                end
            end
        end 

        -- End of secret else block
        end

    end

    -- Note: we just don't do anything with expanded_idx. It's basically a depricated value in these, but it's easier to leave it in
    return reverse_lookup, active_idx, expanded_idx, edges_interior, edges_frontier
end

-- Given importance predictions, do graph update
function Graph:updateGraphFromImportance(importance_orig, reverse_lookup, active_idx, expanded_idx, edges, edge_conf, num_expand)
    -- Make copy of importance 
    local importance = torch.Tensor(importance_orig:size()):copy(importance_orig)

    -- First, minus out all of the already expanded nodes
    local num_left = #active_idx - #expanded_idx
    if num_left == 0 then
        -- Nothing to do, all nodes expanded
        -- DEBUG
        --print('All nodes explanded!')
        return reverse_lookup, active_idx, expanded_idx, edges, edge_conf
    end
    for i = 1, #expanded_idx do
        local node_idx = expanded_idx[i]
        local temp_idx = reverse_lookup[node_idx]
        importance[temp_idx] = -1
    end

    -- Get top important nodes and add to expand lists
    local to_expand_idx = {} 
    for i = 1, math.min(num_expand, num_left) do
        local m, j = torch.max(importance, 1)
        -- Max importance
        local subgraph_idx = j[1]
        local real_idx = active_idx[subgraph_idx]
        table.insert(expanded_idx, real_idx) 
        table.insert(to_expand_idx, real_idx)
        importance[subgraph_idx] = -1
    end
    
    -- Now expand graph
    reverse_lookup, active_idx, edges, edge_conf = self:getExpandedGraph(reverse_lookup, active_idx, to_expand_idx, edges, edge_conf)
    
    return reverse_lookup, active_idx, expanded_idx, edges, edge_conf
end

-- Given importance predictions, do graph update
-- This version uses importance input and just checks if each node should be expanded or not
function Graph:updateGraphFromImportanceSelection(importance_orig, reverse_lookup, active_idx, expanded_idx, edges, edge_conf)
    -- Make copy of importance 
    local importance = torch.Tensor(importance_orig:size()):copy(importance_orig)

    -- First, minus out all of the already expanded nodes
    local num_left = #active_idx - #expanded_idx
    if num_left == 0 then
        -- Nothing to do, all nodes expanded
        -- DEBUG
        --print('All nodes explanded!')
        return reverse_lookup, active_idx, expanded_idx, edges, edge_conf
    end
    for i = 1, #expanded_idx do
        local node_idx = expanded_idx[i]
        local temp_idx = reverse_lookup[node_idx]
        importance[temp_idx] = -1
    end

    -- Now select all the nodes above 0.5 (ones that have been selected)
    local to_expand_idx = {} 
    for i = 1, importance:size(1) do
        if importance[i] > 0.5 then
            local real_idx = active_idx[i]
            table.insert(expanded_idx, real_idx)
            table.insert(to_expand_idx, real_idx)
        end
    end
    
    -- Now expand graph
    reverse_lookup, active_idx, edges, edge_conf = self:getExpandedGraph(reverse_lookup, active_idx, to_expand_idx, edges, edge_conf)
    
    return reverse_lookup, active_idx, expanded_idx, edges, edge_conf
end

-- Given importance predictions, do graph update
function Graph:updateGraphFromAttention(importance_orig, reverse_lookup, active_idx, expanded_idx, edges_interior, edges_frontier, num_expand)
    -- Make copy of importance 
    local importance = torch.Tensor(importance_orig:size()):copy(importance_orig)

    -- First, minus out all of the already expanded nodes
    local num_left = #active_idx - #expanded_idx
    if num_left == 0 then
        -- Nothing to do, all nodes expanded
        -- DEBUG
        --print('All nodes explanded!')
        return reverse_lookup, active_idx, expanded_idx, edges_interior, edges_frontier
    end
    for i = 1, #expanded_idx do
        local node_idx = expanded_idx[i]
        local temp_idx = reverse_lookup[node_idx]
        importance[temp_idx] = -1
    end

    -- Get top important nodes and add to expand lists
    local to_expand_idx = {} 
    for i = 1, math.min(num_expand, num_left) do
        local m, j = torch.max(importance, 1)
        -- Max importance
        local subgraph_idx = j[1]
        local real_idx = active_idx[subgraph_idx]
        table.insert(expanded_idx, real_idx)
        table.insert(to_expand_idx, real_idx)
        importance[subgraph_idx] = -1
    end
    
    -- Now expand graph
    reverse_lookup, active_idx, edges_interior, edge_frontier = self:getExpandedGraphWA(reverse_lookup, active_idx, to_expand_idx, edges_interior, edges_frontier)
    
    return reverse_lookup, active_idx, expanded_idx, edges_interior, edges_frontier
end


-- Given reverse lookup table for graph, currently active nodes and list of nodes to expand, return new active nodes, reverse lookup table, and edge list
function Graph:getExpandedGraph(reverse_lookup, active_idx, expand_idx, edges, edge_conf)
    local n_active = #active_idx

    -- Loop over all the expand nodes
    for i = 1, #expand_idx do
        local toexp_idx = expand_idx[i]
        local toexp_node = self.nodes[toexp_idx]
        
        -- Look at all incoming and outgoing nodes
        for j = 1, #toexp_node.outgoing_edges do
            local cur_edge = toexp_node.outgoing_edges[j]
            local cur_node = cur_edge.end_node
            local cur_idx = cur_node.index
        
            -- If not in graph, add to graph 
            if reverse_lookup[cur_idx] == -1 then
                table.insert(active_idx, cur_idx)
                n_active = n_active + 1
                reverse_lookup[cur_idx] = n_active                

                -- Go through edges to new node and add them (if they are in our subgraph)
                for k = 1, #cur_node.outgoing_edges do
                    local add_edge = cur_node.outgoing_edges[k]
                    local other_node_idx = add_edge.end_node.index
                    
                    -- If other end of edge is in the graph already, add the edge
                    if reverse_lookup[other_node_idx] ~= -1 then
                        local newedge = {reverse_lookup[cur_idx], add_edge.edgetype.index, reverse_lookup[other_node_idx]}
                        table.insert(edges, newedge)
                        table.insert(edge_conf, add_edge.confidence)
                    end
                end
                for k = 1, #cur_node.incoming_edges do
                    local add_edge = cur_node.incoming_edges[k]
                    local other_node_idx = add_edge.start_node.index
                    
                    -- If other end of edge is in the graph already, add the edge
                    if reverse_lookup[other_node_idx] ~= -1 then
                        local newedge = {reverse_lookup[other_node_idx], add_edge.edgetype.index, reverse_lookup[cur_idx]}
                        table.insert(edges, newedge)
                        table.insert(edge_conf, add_edge.confidence)
                    end 
                end
            end
        end
        for j = 1, #toexp_node.incoming_edges do
            local cur_node = toexp_node.incoming_edges[j].start_node
            local cur_idx = cur_node.index
            if reverse_lookup[cur_idx] == -1 then
                table.insert(active_idx, cur_idx)
                n_active = n_active + 1
                reverse_lookup[cur_idx] = n_active                    
       
                -- Go through edges to new node and add them (if they are in our subgraph)
                for k = 1, #cur_node.outgoing_edges do
                    local add_edge = cur_node.outgoing_edges[k]
                    local other_node_idx = add_edge.end_node.index
                    
                    -- If other end of edge is in the graph already, add the edge
                    if reverse_lookup[other_node_idx] ~= -1 then
                        local newedge = {reverse_lookup[cur_idx], add_edge.edgetype.index, reverse_lookup[other_node_idx]}
                        table.insert(edges, newedge)
                        table.insert(edge_conf, add_edge.confidence)
                    end
                end
                for k = 1, #cur_node.incoming_edges do
                    local add_edge = cur_node.incoming_edges[k]
                    local other_node_idx = add_edge.start_node.index
                    
                    -- If other end of edge is in the graph already, add the edge
                    if reverse_lookup[other_node_idx] ~= -1 then
                        local newedge = {reverse_lookup[other_node_idx], add_edge.edgetype.index, reverse_lookup[cur_idx]}
                        table.insert(edges, newedge)
                        table.insert(edge_conf, add_edge.confidence)
                    end
                end
            end
        end
    end

    -- Return new values
    return reverse_lookup, active_idx, edges, edge_conf
end

-- Same as before, but has to update edges_interior and edges_frontier lists
-- Given reverse lookup table for graph, currently active nodes and list of nodes to expand, return new active nodes, reverse lookup table, and edge list
function Graph:getExpandedGraphWA(reverse_lookup, active_idx, expand_idx, edges_interior, edges_frontier)
    local n_active = #active_idx

    -- Loop over all the expand nodes
    for i = 1, #expand_idx do
        local toexp_idx = expand_idx[i]
        local toexp_node = self.nodes[toexp_idx]
        
        -- Look at all incoming and outgoing nodes
        for j = 1, #toexp_node.outgoing_edges do
            local cur_edge = toexp_node.outgoing_edges[j]
            local cur_node = cur_edge.end_node
            local cur_idx = cur_node.index
        
            -- If not in graph, add to graph 
            if reverse_lookup[cur_idx] == -1 then
                table.insert(active_idx, cur_idx)
                n_active = n_active + 1
                reverse_lookup[cur_idx] = n_active                

                -- Go through edges to new node and add them (if they are in our subgraph)
                for k = 1, #cur_node.outgoing_edges do
                    local add_edge = cur_node.outgoing_edges[k]
                    local other_node_idx = add_edge.end_node.index
                    local edge_struct = {}
                    edge_struct.edge = {reverse_lookup[cur_idx], add_edge.edgetype.index, reverse_lookup[other_node_idx]}
                    edge_struct.id = add_edge.index

                    -- If other node is not in graph, add the edge to edges_frontier
                    if reverse_lookup[other_node_idx] == -1 then
                        edges_frontier[add_edge.index] = edge_struct
                    -- If other end of edges is in graph already, remove edge from edges_frontier and add to edges_interior
                    else
                        edges_frontier[add_edge.index] = nil
                        table.insert(edges_interior, edge_struct)
                        --print('Insert 1')
                    end
                end
                for k = 1, #cur_node.incoming_edges do
                    local add_edge = cur_node.incoming_edges[k]
                    local other_node_idx = add_edge.start_node.index
                    local edge_struct = {}
                    edge_struct.edge = {reverse_lookup[other_node_idx], add_edge.edgetype.index, reverse_lookup[cur_idx]}
                    edge_struct.id = add_edge.index

                    -- If other node is not in graph, add the edge to edges_frontier
                    if reverse_lookup[other_node_idx] == -1 then
                        edges_frontier[add_edge.index] = edge_struct
                    -- If other end of edges is in graph already, remove edge from edges_frontier and add to edges_interior
                    else
                        edges_frontier[add_edge.index] = nil
                        table.insert(edges_interior, edge_struct)
                        --print('Insert 2')
                    end 
                end
            end
        end
        for j = 1, #toexp_node.incoming_edges do
            local cur_node = toexp_node.incoming_edges[j].start_node
            local cur_idx = cur_node.index
            if reverse_lookup[cur_idx] == -1 then
                table.insert(active_idx, cur_idx)
                n_active = n_active + 1
                reverse_lookup[cur_idx] = n_active                    
       
                -- Go through edges to new node and add them (if they are in our subgraph)
                for k = 1, #cur_node.outgoing_edges do
                    local add_edge = cur_node.outgoing_edges[k]
                    local other_node_idx = add_edge.end_node.index
                    local edge_struct = {}
                    edge_struct.edge = {reverse_lookup[cur_idx], add_edge.edgetype.index, reverse_lookup[other_node_idx]}
                    edge_struct.id = add_edge.index                   

                    -- If other node is not in graph, add the edge to edges_frontier
                    if reverse_lookup[other_node_idx] == -1 then
                        edges_frontier[add_edge.index] = edge_struct
                    -- If other end of edges is in graph already, remove edge from edges_frontier and add to edges_interior
                    else
                        edges_frontier[add_edge.index] = nil
                        table.insert(edges_interior, edge_struct)
                        --print('Insert 3')
                    end
                end
                for k = 1, #cur_node.incoming_edges do
                    local add_edge = cur_node.incoming_edges[k]
                    local other_node_idx = add_edge.start_node.index
                    local edge_struct = {}
                    edge_struct.edge = {reverse_lookup[other_node_idx], add_edge.edgetype.index, reverse_lookup[cur_idx]}
                    edge_struct.id = add_edge.index

                    -- If other node is not in graph, add the edge to edges_frontier
                    if reverse_lookup[other_node_idx] == -1 then
                        edges_frontier[add_edge.index] = edge_struct
                    -- If other end of edges is in graph already, remove edge from edges_frontier and add to edges_interior
                    else
                        --edges_frontier[add_edge.index] = nil
                        table.insert(edges_interior, edge_struct)
                        --print('Insert 4')
                    end
                end
            end
        end
    end

    -- Return new values
    return reverse_lookup, active_idx, edges_interior, edges_frontier
end

-- Given target nodes in that should be 1, propogates value to all nodes in graph
-- Returns value for each node
function Graph:getDiscountedValues(vocab_target_idx, gamma, num_steps)
    local node_values = torch.Tensor(self.n_total_nodes):zero()
    local visited = torch.ByteTensor(self.n_total_nodes):zero()

    -- First get graph indices and set targets to 1
    local frontier = {}
    for i = 1, #vocab_target_idx do
        local cur_vocab_idx = vocab_target_idx[i]
        local graph_idx = self.vocab_reverse_lookup[cur_vocab_idx]
        if graph_idx ~= -1 then
            node_values[graph_idx] = 1
            visited[graph_idx] = 1
            table.insert(frontier, graph_idx)
        end
    end

    -- Loop over steps and set discounted rewards
    local value = 1
    
    -- Loop over steps
    for step = 1, num_steps do
        value = value * gamma
        local new_frontier = {}
     
        -- Look at nodes on frontier, and set values for their neighbors
        for i = 1, #frontier do
            local front_node_idx = frontier[i]
            local front_node = self.nodes[front_node_idx]
            for j = 1, #front_node.outgoing_edges do
               local edge = front_node.outgoing_edges[j]
               local other_node_idx = edge.end_node.index 
               if visited[other_node_idx] == 0 then
                   assert(node_values[other_node_idx] == 0)
                   table.insert(new_frontier, other_node_idx)
                   node_values[other_node_idx] = value
                   visited[other_node_idx] = 1
               end
            end
            for j = 1, #front_node.incoming_edges do
                local edge = front_node.incoming_edges[j]
                local other_node_idx = edge.start_node.index
                if visited[other_node_idx] == 0 then
                    assert(node_values[other_node_idx] == 0)
                    table.insert(new_frontier, other_node_idx)
                    node_values[other_node_idx] = value
                    visited[other_node_idx] = 1
                end
            end
        end
        frontier = new_frontier

        -- If all nodes have been visited, or the nodes are not discoverable
        -- then we're done
        if #frontier == 0 then
            break
        end
    end
    return node_values
end

-- Given target nodes in that should be 1, propogates value to all nodes in graph
-- Returns value for each node
function Graph:getDiscountedValuesGraphInds(graph_node_idx, gamma, num_steps)
    local node_values = torch.Tensor(self.n_total_nodes):zero()
    local visited = torch.ByteTensor(self.n_total_nodes):zero()

    -- First get graph indices and set targets to 1
    local frontier = {}
    for i = 1, #graph_node_idx do
        local graph_idx = graph_node_idx[i]
        if graph_idx ~= -1 then
            node_values[graph_idx] = 1
            visited[graph_idx] = 1
            table.insert(frontier, graph_idx)
        end
    end

    -- Loop over steps and set discounted rewards
    local value = 1
    
    -- Loop over steps
    for step = 1, num_steps do
        value = value * gamma
        local new_frontier = {}
     
        -- Look at nodes on frontier, and set values for their neighbors
        for i = 1, #frontier do
            local front_node_idx = frontier[i]
            local front_node = self.nodes[front_node_idx]
            for j = 1, #front_node.outgoing_edges do
               local edge = front_node.outgoing_edges[j]
               local other_node_idx = edge.end_node.index 
               if visited[other_node_idx] == 0 then
                   assert(node_values[other_node_idx] == 0)
                   table.insert(new_frontier, other_node_idx)
                   node_values[other_node_idx] = value
                   visited[other_node_idx] = 1
               end
            end
            for j = 1, #front_node.incoming_edges do
                local edge = front_node.incoming_edges[j]
                local other_node_idx = edge.start_node.index
                if visited[other_node_idx] == 0 then
                    assert(node_values[other_node_idx] == 0)
                    table.insert(new_frontier, other_node_idx)
                    node_values[other_node_idx] = value
                    visited[other_node_idx] = 1
                end
            end
        end
        frontier = new_frontier

        -- If all nodes have been visited, or the nodes are not discoverable
        -- then we're done
        if #frontier == 0 then
            break
        end
    end
    return node_values
end
