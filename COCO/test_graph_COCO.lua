-- Baseline model to train ~300 classification on Visual Genome
-- Basically, VGGNet that has output for all the Visual Genome classes we care about
require 'torch'
require 'optim'
require 'nn'
require 'cunn'
require 'cudnn'
require '../gsnn'

opt = {
    dataloc = '/mnt/disk2/kennethm/VisualGenome/COCO/',  -- Base directory of data
    imageloc = '/mnt/disk2/kennethm/coco/images/',       -- Base directory of images
    load_net = '/mnt/disk1/kennethm/graphmodels_coco/graph_vgonly_COCO_multiclass_net_epoch_20.t7',                      -- Classification net to load
    load_vgg_net = '/mnt/disk1/kennethm/graphmodels_coco/graph_vgonly_COCO_vgg_net_epoch_20.t7',                  -- VGG net to load
    load_gsnn_net = '/mnt/disk1/kennethm/graphmodels_coco/graph_vgonly_COCO_gsnn_net_epoch_20.t7',                 -- GSNN net to load
    load_opt = '/mnt/disk1/kennethm/graphmodels_coco/graph_vgonly_COCO_params',  -- Where to load options used to train the net
    outputdir = '/mnt/disk1/kennethm/graphmodels_coco/', -- Output directory
    batchSize = 32,                     -- Image batch sizez
    nThreads = 0,                       -- Number of threads
    gpu = 1,                            -- Which GPU to use
}
 
-- Argument parser
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

torch.setdefaulttensortype('torch.FloatTensor')

-- Timers
local data_tm = torch.Timer()
local tm = torch.Timer()
local graph_tm = torch.Timer()

-- Load training opt
train_opt = torch.load(opt.load_opt)
if train_opt.fc7_to_ann == nil then
    train_opt.fc7_to_ann = 0
end
if train_opt.graph_only == nil then
     train_opt.graph_only = 0
end

-- Set opt parameters needed in data
opt.manualSeed = train_opt.manualSeed
opt.runmode = train_opt.runmode
opt.detectMode = train_opt.detectMode
assert(opt.detectMode == 'coco')
opt.loadSize = train_opt.loadSize
opt.fineSize = train_opt.fineSize

-- Initialize data
local DataLoader = paths.dofile('data/data.lua')
local data = DataLoader.new(opt.nThreads, opt)
print('Dataset test size: ', data:testSize())

-- Load graph
local graph = torch.load(opt.dataloc .. '/torchdata/pruned_graph_cocosplit.t7')[1]

-- Set GPU and load nets
-- Note: either start with VGGNet or resume previous model
cutorch.setDevice(opt.gpu)
local fc7_size = 4096
local output_size = 80
local detect_size = 80
local graph_size = train_opt.context_dim * output_size
local net = torch.load(opt.load_net)
net:cuda()
net:evaluate()
local vgg_net = torch.load(opt.load_vgg_net)
vgg_net:cuda()
vgg_net:evaluate()
local gsnn_net = gsnn.load_gsnn_from_file(opt.load_gsnn_net)
local gsnn_nets = {}
gsnn_nets[1] = gsnn_net
for i = 2, opt.batchSize do
    gsnn_nets[i] = gsnn_net:create_share_param_copy()
end
collectgarbage(); collectgarbage()
print('Loaded nets')

-- Criterions
local criterion = nn.BCECriterion()

-- Get thresholds
local numthresholds = 101
local midpoint = math.floor(numthresholds/2)
local thresholds = torch.Tensor(numthresholds):zero()
for i = 1, numthresholds do
    thresholds[i] = (i-1) / (numthresholds-1)    
end

-- Losses
local multiclass_loss = 0
local importance_loss = 0
local total_multiclass_loss = 0
local total_precision_thresh = torch.Tensor(thresholds:size()):zero()
local total_recall_thresh = torch.Tensor(thresholds:size()):zero()
local total_precision_obj = torch.Tensor(thresholds:size()):zero()
local total_recall_obj = torch.Tensor(thresholds:size()):zero()
local total_precision_att = torch.Tensor(thresholds:size()):zero()
local total_recall_att = torch.Tensor(thresholds:size()):zero()
local total_precision_coco = torch.Tensor(thresholds:size()):zero()
local total_recall_coco = torch.Tensor(thresholds:size()):zero()
local total_precision_topobj = torch.Tensor(thresholds:size()):zero()
local total_recall_topobj = torch.Tensor(thresholds:size()):zero()
local total_precision_botobj = torch.Tensor(thresholds:size()):zero()
local total_recall_botobj = torch.Tensor(thresholds:size()):zero()
local total_precision_topatt = torch.Tensor(thresholds:size()):zero()
local total_recall_topatt = torch.Tensor(thresholds:size()):zero()
local total_precision_botatt = torch.Tensor(thresholds:size()):zero()
local total_recall_botatt = torch.Tensor(thresholds:size()):zero()
local count = 0

-- Load cuda data
local image_data = torch.Tensor(opt.batchSize, 3, train_opt.fineSize, train_opt.fineSize)
local label = torch.Tensor(opt.batchSize, output_size)
local detect_annotation = torch.Tensor(opt.batchSize, detect_size)
local graph_data = torch.Tensor(opt.batchSize, graph_size)
image_data = image_data:cuda()
label = label:cuda()
detect_annotation = detect_annotation:cuda()
graph_data = graph_data:cuda()
criterion:cuda()

print('GPU Data initialized')
collectgarbage(); collectgarbage()

-- Get subdivisions of data to report
local listdata = torch.load(opt.dataloc .. '/torchdata/torchlists.t7')
local objects = listdata[1]
local attributes = listdata[2]
local relationships = listdata[3]
local coco_cats = listdata[4]
local is_object = torch.Tensor(output_size):zero()
local is_attribute = torch.Tensor(output_size):zero()
local is_coco = torch.Tensor(output_size):zero()
local is_top_object = torch.Tensor(output_size):zero()
local is_bottom_object = torch.Tensor(output_size):zero()
local is_top_attribute = torch.Tensor(output_size):zero()
local is_bottom_attribute = torch.Tensor(output_size):zero()
for i = 1, output_size do
    if i <= #objects then
        is_object[i] = 1
        if i <= 10 then
            is_top_object[i] = 1
        elseif #objects - i < 10 then
            is_bottom_object[i] = 1
        end
        for j = 1, #coco_cats do
            if objects[i] == coco_cats[j] then
                is_coco[i] = 1
            end 
        end
    else
        is_attribute[i] = 1
        if i - #objects <= 10 then
            is_top_attribute[i] = 1
        elseif (#objects+#attributes) - i < 10 then
            is_bottom_attribute[i] = 1
        end
    end    
end
local obj_mask = torch.repeatTensor(is_object, opt.batchSize, 1)
local att_mask = torch.repeatTensor(is_attribute, opt.batchSize, 1)
local coco_mask = torch.repeatTensor(is_coco, opt.batchSize, 1)
local topobj_mask = torch.repeatTensor(is_top_object, opt.batchSize, 1)
local botobj_mask = torch.repeatTensor(is_bottom_object, opt.batchSize, 1)
local topatt_mask = torch.repeatTensor(is_top_attribute, opt.batchSize, 1)
local botatt_mask = torch.repeatTensor(is_bottom_attribute, opt.batchSize, 1)

local outputs = {}

-- Testing loop
local numtest = math.floor(data:testSize())
for i = 1, numtest, opt.batchSize do
    tm:reset()

    -- Get test data
    data_tm:reset(); data_tm:resume()
    local imdata, detectConf, detectHid, detectClass, detectionTable, lab, labinds = data:getTestBatch(i)
    data_tm:stop() 

    -- Check if we've reached end
    if imdata == nil then
        break
    end

    -- Copy data
    image_data:copy(imdata)
    local label_cpu = lab
    label:copy(lab)
    detectConf = torch.reshape(detectConf, opt.batchSize, detect_size)
    detect_annotation:copy(detectConf)

    -- Forward through VGG
    local vgg_out = vgg_net:forward(image_data)

    -- Forward through GSNN
    local graph_data_cpu = torch.Tensor(graph_data:size()):zero()
    for i = 1, opt.batchSize do
        local batchAnnotation = torch.reshape(detectConf[i], detect_size)
        local initial_conf = batchAnnotation
        local annotations_plus = torch.Tensor(detect_size, 1):zero() -- Hack
    
        -- Forward through GSNN network
        local output, importance_outputs, reverse_lookup, active_idx, expanded_idx = gsnn_nets[i]:forward(graph, initial_conf, annotations_plus)
        
        -- Reorder output of graph
        for j = 1, #active_idx do
            -- Get vocab index
            local full_graph_idx = active_idx[j]
            local output_idx = graph.nodes[full_graph_idx].neil_detector_index

            -- Set correct part of graph_data_cpu
            if output_idx ~= -1 then
                assert(output_idx <= 80)
                graph_data_cpu[{{i}, {(output_idx-1)*train_opt.context_dim+1, output_idx*train_opt.context_dim}}] = output[j]
            end
        end
    end
    graph_data:copy(graph_data_cpu)
    
    -- Forward pass
    local output = net:forward({vgg_out, detect_annotation, graph_data})

    -- Save output
    local output_cpu = torch.Tensor(output:size()):zero()
    output_cpu:copy(output)
    table.insert(outputs, output_cpu)

    -- Get multiclass error
    local out_cpu = torch.Tensor(output:size())
    multiclass_loss = criterion:forward(output, label)
    total_multiclass_loss = total_multiclass_loss + multiclass_loss  

    -- Get precision and recall
    for j = 1, thresholds:size(1) do
        local thresh = thresholds[j]
        
        -- Get the masks for predicted positives, label positive and their intersection
        local pred_positives = torch.Tensor(output:size())
        pred_positives:copy(output)
        pred_positives[torch.gt(pred_positives, thresh)] = 1
        pred_positives[torch.le(pred_positives, thresh)] = 0
        local pred_positives_obj = torch.cmul(pred_positives, obj_mask)
        local pred_positives_att = torch.cmul(pred_positives, att_mask)
        local pred_positives_coco = torch.cmul(pred_positives, coco_mask)
        local pred_positives_topobj = torch.cmul(pred_positives, topobj_mask)
        local pred_positives_botobj = torch.cmul(pred_positives, botobj_mask)
        local pred_positives_topatt = torch.cmul(pred_positives, topatt_mask)
        local pred_positives_botatt = torch.cmul(pred_positives, botatt_mask) 
        local label_positives = torch.Tensor(lab:size())
        label_positives:copy(lab)    
        local label_positives_obj = torch.cmul(label_positives, obj_mask)
        local label_positives_att = torch.cmul(label_positives, att_mask)
        local label_positives_coco = torch.cmul(label_positives, coco_mask)
        local label_positives_topobj = torch.cmul(label_positives, topobj_mask)  
        local label_positives_botobj = torch.cmul(label_positives, botobj_mask)
        local label_positives_topatt = torch.cmul(label_positives, topatt_mask)
        local label_positives_botatt = torch.cmul(label_positives, botatt_mask)
        local intersection = torch.cmul(pred_positives, label_positives)
        local intersection_obj = torch.cmul(pred_positives_obj, label_positives_obj)
        local intersection_att = torch.cmul(pred_positives_att, label_positives_att)
        local intersection_coco = torch.cmul(pred_positives_coco, label_positives_coco)
        local intersection_topobj = torch.cmul(pred_positives_topobj, label_positives_topobj)
        local intersection_botobj = torch.cmul(pred_positives_botobj, label_positives_botobj)
        local intersection_topatt = torch.cmul(pred_positives_topatt, label_positives_topatt)
        local intersection_botatt = torch.cmul(pred_positives_botatt, label_positives_botatt)

        -- Sum up intersection, retrieved and relevant
        local num_intersection = torch.cumsum(intersection, 2):narrow(2, intersection:size(2), 1)
        local num_intersection_obj = torch.cumsum(intersection_obj, 2):narrow(2, intersection_obj:size(2), 1)
        local num_intersection_att = torch.cumsum(intersection_att, 2):narrow(2, intersection_att:size(2), 1)
        local num_intersection_coco = torch.cumsum(intersection_coco, 2):narrow(2, intersection_coco:size(2), 1)
        local num_intersection_topobj = torch.cumsum(intersection_topobj, 2):narrow(2, intersection_topobj:size(2), 1)
        local num_intersection_botobj = torch.cumsum(intersection_botobj, 2):narrow(2, intersection_topobj:size(2), 1)
        local num_intersection_topatt = torch.cumsum(intersection_topatt, 2):narrow(2, intersection_topatt:size(2), 1)
        local num_intersection_botatt = torch.cumsum(intersection_botatt, 2):narrow(2, intersection_botatt:size(2), 1)
        local num_retrieved = torch.cumsum(pred_positives, 2):narrow(2, pred_positives:size(2), 1)
        local num_retrieved_obj = torch.cumsum(pred_positives_obj, 2):narrow(2, pred_positives_obj:size(2), 1)
        local num_retrieved_att = torch.cumsum(pred_positives_att, 2):narrow(2, pred_positives_att:size(2), 1)
        local num_retrieved_coco = torch.cumsum(pred_positives_coco, 2):narrow(2, pred_positives_coco:size(2), 1) 
        local num_retrieved_topobj = torch.cumsum(pred_positives_topobj, 2):narrow(2, pred_positives_topobj:size(2), 1)
        local num_retrieved_botobj = torch.cumsum(pred_positives_botobj, 2):narrow(2, pred_positives_botobj:size(2), 1)
        local num_retrieved_topatt = torch.cumsum(pred_positives_topatt, 2):narrow(2, pred_positives_topatt:size(2), 1)
        local num_retrieved_botatt = torch.cumsum(pred_positives_botatt, 2):narrow(2, pred_positives_botatt:size(2), 1)
        local num_relevant = torch.cumsum(label_positives, 2):narrow(2, label_positives:size(2), 1)
        local num_relevant_obj = torch.cumsum(label_positives_obj, 2):narrow(2, label_positives_obj:size(2), 1)
        local num_relevant_att = torch.cumsum(label_positives_att, 2):narrow(2, label_positives_att:size(2), 1)
        local num_relevant_coco = torch.cumsum(label_positives_coco, 2):narrow(2, label_positives_coco:size(2), 1)
        local num_relevant_topobj = torch.cumsum(label_positives_topobj, 2):narrow(2, label_positives_topobj:size(2), 1)
        local num_relevant_botobj = torch.cumsum(label_positives_botobj, 2):narrow(2, label_positives_botobj:size(2), 1)
        local num_relevant_topatt = torch.cumsum(label_positives_topatt, 2):narrow(2, label_positives_topatt:size(2), 1)
        local num_relevant_botatt = torch.cumsum(label_positives_botatt, 2):narrow(2, label_positives_botatt:size(2), 1)

        -- Calculate and sum precision and recall
        local precision = torch.Tensor(num_intersection:size())
        precision:copy(num_intersection:cdiv(num_retrieved))
        precision[torch.eq(num_retrieved, 0)] = 1
        num_intersection = torch.cumsum(intersection, 2):narrow(2, intersection:size(2), 1)
        local recall = torch.Tensor(num_intersection:size())
        recall:copy(num_intersection:cdiv(num_relevant))
        recall[torch.eq(num_relevant, 0)] = 1
        num_intersection = torch.cumsum(intersection, 2):narrow(2, intersection:size(2), 1)
        local precision_obj = torch.Tensor(num_intersection_obj:size())
        precision_obj:copy(num_intersection_obj:cdiv(num_retrieved_obj))
        precision_obj[torch.eq(num_retrieved_obj, 0)] = 1
        num_intersection_obj = torch.cumsum(intersection_obj, 2):narrow(2, intersection_obj:size(2), 1)
        local recall_obj = torch.Tensor(num_intersection_obj:size())
        recall_obj:copy(num_intersection_obj:cdiv(num_relevant_obj))
        recall_obj[torch.eq(num_relevant_obj, 0)] = 1
        num_intersection_obj = torch.cumsum(intersection_obj, 2):narrow(2, intersection_obj:size(2), 1) 
        local precision_att = torch.Tensor(num_intersection_att:size())
        precision_att:copy(num_intersection_att:cdiv(num_retrieved_att))
        precision_att[torch.eq(num_retrieved_att, 0)] = 1
        num_intersection_att = torch.cumsum(intersection_att, 2):narrow(2, intersection_att:size(2), 1)
        local recall_att = torch.Tensor(num_intersection_att:size())
        recall_att:copy(num_intersection_att:cdiv(num_relevant_att))
        recall_att[torch.eq(num_relevant_att, 0)] = 1
        num_intersection_att = torch.cumsum(intersection_att, 2):narrow(2, intersection_att:size(2), 1) 
        local precision_coco = torch.Tensor(num_intersection_coco:size())
        precision_coco:copy(num_intersection_coco:cdiv(num_retrieved_coco))
        precision_coco[torch.eq(num_retrieved_coco, 0)] = 1
        num_intersection_coco = torch.cumsum(intersection_coco, 2):narrow(2, intersection_coco:size(2), 1)
        local recall_coco = torch.Tensor(num_intersection_coco:size())
        recall_coco:copy(num_intersection_coco:cdiv(num_relevant_coco))
        recall_coco[torch.eq(num_relevant_coco, 0)] = 1
        num_intersection_coco = torch.cumsum(intersection_coco, 2):narrow(2, intersection_coco:size(2), 1)
        local precision_topobj = torch.Tensor(num_intersection_topobj:size())
        precision_topobj:copy(num_intersection_topobj:cdiv(num_retrieved_topobj))
        precision_topobj[torch.eq(num_retrieved_topobj, 0)] = 1
        num_intersection_topobj = torch.cumsum(intersection_topobj, 2):narrow(2, intersection_topobj:size(2), 1)
        local recall_topobj = torch.Tensor(num_intersection_topobj:size())
        recall_topobj:copy(num_intersection_topobj:cdiv(num_relevant_topobj))
        recall_topobj[torch.eq(num_relevant_topobj, 0)] = 1
        num_intersection_topobj = torch.cumsum(intersection_topobj, 2):narrow(2, intersection_topobj:size(2), 1) 
        local precision_botobj = torch.Tensor(num_intersection_botobj:size())
        precision_botobj:copy(num_intersection_botobj:cdiv(num_retrieved_botobj))
        precision_botobj[torch.eq(num_retrieved_botobj, 0)] = 1
        num_intersection_botobj = torch.cumsum(intersection_botobj, 2):narrow(2, intersection_botobj:size(2), 1)
        local recall_botobj = torch.Tensor(num_intersection_botobj:size())
        recall_botobj:copy(num_intersection_botobj:cdiv(num_relevant_botobj))
        recall_botobj[torch.eq(num_relevant_botobj, 0)] = 1
        num_intersection_botobj = torch.cumsum(intersection_botobj, 2):narrow(2, intersection_botobj:size(2), 1) 
        local precision_topatt = torch.Tensor(num_intersection_topatt:size())
        precision_topatt:copy(num_intersection_topatt:cdiv(num_retrieved_topatt))
        precision_topatt[torch.eq(num_retrieved_topatt, 0)] = 1
        num_intersection_topatt = torch.cumsum(intersection_topatt, 2):narrow(2, intersection_topatt:size(2), 1)
        local recall_topatt = torch.Tensor(num_intersection_topatt:size())
        recall_topatt:copy(num_intersection_topatt:cdiv(num_relevant_topatt))
        recall_topatt[torch.eq(num_relevant_topatt, 0)] = 1
        num_intersection_topatt = torch.cumsum(intersection_topatt, 2):narrow(2, intersection_topatt:size(2), 1) 
        local precision_botatt = torch.Tensor(num_intersection_botatt:size())
        precision_botatt:copy(num_intersection_botatt:cdiv(num_retrieved_botatt))
        precision_botatt[torch.eq(num_retrieved_botatt, 0)] = 1
        num_intersection_botatt = torch.cumsum(intersection_botatt, 2):narrow(2, intersection_botatt:size(2), 1)
        local recall_botatt = torch.Tensor(num_intersection_botatt:size())
        recall_botatt:copy(num_intersection_botatt:cdiv(num_relevant_botatt))
        recall_botatt[torch.eq(num_relevant_botatt, 0)] = 1
        num_intersection_botatt = torch.cumsum(intersection_botatt, 2):narrow(2, intersection_botatt:size(2), 1) 
        
        -- Calculate totals
        total_precision_thresh[j] = total_precision_thresh[j] + precision:sum()
        total_recall_thresh[j] = total_recall_thresh[j] + recall:sum()
        total_precision_obj[j] = total_precision_obj[j] + precision_obj:sum()
        total_recall_obj[j] = total_recall_obj[j] + recall_obj:sum()
        total_precision_att[j] = total_precision_att[j] + precision_att:sum()
        total_recall_att[j] = total_recall_att[j] + recall_att:sum()
        total_precision_coco[j] = total_precision_coco[j] + precision_coco:sum()
        total_recall_coco[j] = total_recall_coco[j] + recall_coco:sum()
        total_precision_topobj[j] = total_precision_topobj[j] + precision_topobj:sum()
        total_recall_topobj[j] = total_recall_topobj[j] + recall_topobj:sum()
        total_precision_botobj[j] = total_precision_botobj[j] + precision_botobj:sum()
        total_recall_botobj[j] = total_recall_botobj[j] + recall_botobj:sum()
        total_precision_topatt[j] = total_precision_topatt[j] + precision_topatt:sum()
        total_recall_topatt[j] = total_recall_topatt[j] + recall_topatt:sum()
        total_precision_botatt[j] = total_precision_botatt[j] + precision_botatt:sum()
        total_recall_botatt[j] = total_recall_botatt[j] + recall_botatt:sum()
    end
    count = count + opt.batchSize     

    print(('Test: [%8d / %8d]\t Time: %.3f  DataTime: %.3f  '
           .. 'Multiclass loss: %.4f  '
           .. 'Prec: %.4f  Recall: %.4f  Obj Prec: %.4f  Obj Recall: %.4f  '
           .. 'Att Prec: %.4f  Att Recall: %.4f  Coco Prec: %.4f  Coco Recall: %.4f  '
           .. 'Top Obj Prec: %.4f  Top Obj Recall: %.4f  Bot Obj Prec: %.4f  Bot Obj Recall: %.4f  '
           .. 'Top Att Prec: %.4f  Top ATt Recall: %.4f  Bot Att Prec: %.4f  Bot Att Recall: %.4f'):format(
           ((i-1)/opt.batchSize), numtest/opt.batchSize, tm:time().real, data_tm:time().real,
           total_multiclass_loss / (count/opt.batchSize), 
           total_precision_thresh[midpoint] / count, total_recall_thresh[midpoint] / count,
           total_precision_obj[midpoint] / count, total_recall_obj[midpoint] / count,
           total_precision_att[midpoint] / count, total_recall_att[midpoint] / count,
           total_precision_coco[midpoint] / count, total_recall_coco[midpoint] / count,
           total_precision_topobj[midpoint] / count, total_recall_topobj[midpoint] / count,
           total_precision_botobj[midpoint] / count, total_recall_botobj[midpoint] / count,
           total_precision_topatt[midpoint] / count, total_recall_topatt[midpoint] / count,
           total_precision_botatt[midpoint] / count, total_recall_botatt[midpoint] / count))
    print('Testing: ' .. train_opt.name)
end

-- Calculate average precision and recall over test set
local avg_precision_thresh = torch.div(total_precision_thresh, count)
local avg_recall_thresh = torch.div(total_recall_thresh, count)
local avg_precision_obj = torch.div(total_precision_obj, count)
local avg_recall_obj = torch.div(total_recall_obj, count)
local avg_precision_att = torch.div(total_precision_att, count)
local avg_recall_att = torch.div(total_recall_att, count)
local avg_precision_coco = torch.div(total_precision_coco, count)
local avg_recall_coco = torch.div(total_recall_coco, count)
local avg_precision_topobj = torch.div(total_precision_topobj, count)
local avg_recall_topobj = torch.div(total_recall_topobj, count)
local avg_precision_botobj = torch.div(total_precision_botobj, count)
local avg_recall_botobj = torch.div(total_recall_botobj, count)
local avg_precision_topatt = torch.div(total_precision_topatt, count)
local avg_recall_topatt = torch.div(total_recall_topatt, count)
local avg_precision_botatt = torch.div(total_precision_botatt, count)
local avg_recall_botatt = torch.div(total_recall_botatt, count)
local avg_multiclass_loss = total_multiclass_loss / (count/opt.batchSize)

-- Save test results
local testdata = {thresholds, avg_precision_thresh, avg_recall_thresh, avg_precision_obj, avg_recall_obj, 
    avg_precision_att, avg_recall_att, avg_precision_coco, avg_recall_coco, 
    avg_precision_topobj, avg_recall_topobj, avg_precision_botobj, avg_recall_botobj, 
    avg_precision_topatt, avg_recall_topatt, avg_precision_botatt, avg_recall_botatt, 
    outputs, avg_multiclass_loss, count}

torch.save(opt.outputdir .. '/' .. train_opt.name .. '_testresults.t7', testdata)

print(('Over %d testing examples, average precision is %.4f, average recall is %.4f, '
       .. 'average object precision is %.4f, average object recall is %.4f, '
       .. 'average attribute precision is %.4f, average attribute recall is %.4f, '
       .. 'average coco precision is %.4f, average coco recall is %.4f'):format(
       count, avg_precision_thresh[midpoint], avg_recall_thresh[midpoint], 
       avg_precision_obj[midpoint], avg_recall_obj[midpoint],
       avg_precision_att[midpoint], avg_recall_att[midpoint],
       avg_precision_coco[midpoint], avg_recall_coco[midpoint]))
