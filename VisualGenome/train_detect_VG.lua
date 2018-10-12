-- Baseline model to train ~300 classification on Visual Genome
-- Basically, VGGNet that has output for all the Visual Genome classes we care about
require 'torch'
require 'optim'
require 'nn'
require 'cunn'
require 'cudnn'
require '../gsnn'
--require 'PosBCECriterion'

opt = {
    dataloc = '/mnt/disk2/kennethm/VisualGenome/',  -- Base directory of data
    load_net = '/mnt/disk2/kennethm/VisualGenome/VGGNet',  -- Base net to start with, or the classification net
    load_net_type = 'VGG',              -- 'VGG', 'baseline', 'detect', 'graph' 
    load_vgg_net = '',                  -- VGG net to load
    outputdir = '/mnt/disk1/kennethm/graphmodels_new/', -- Output directory
    optim = 'sgd',                      -- Optimizer to use
    learning_rate = 5e-2,               -- Initial learning rate
    VGG_lr_weight = 0.1,                -- What to multiply base learning rate to VGG
    momentum = 0.5,                     -- Initial momentum 
    weight_decay = 1e-6,                -- Weight decay
    batchSize = 16,                     -- Image batch sizez
    num_epochs = 20,                    -- Number of epochs to train for
    epoch_start = 1,                    -- Epoch to start at if you are resumming a previous training session
    ntrain = math.huge,                 -- # examples per epoch. math.huge for full dataset
    save_after = 5,                     -- Number of epochs to save permanent model after
    manualSeed = 0,                     -- Random seed
    name = 'detect_VG',               -- Name of experiment
    nThreads = 4,                       -- Number of threads
    log = '/mnt/disk1/kennethm/vgtlog/',  -- log directory
    loadSize = 256,                     -- size to load image in
    fineSize = 224,                     -- Image size into net (224 for VGG)
    gpu = 1,                            -- Which GPU to use
    runmode = 'detect',               -- baseline - just image input, detect - image + detections, graph - image + detections + gsnn
    lossweight = 20,
    gamma = 0.1,                        -- How much to reduce learning rate
    gamma_after = 10,                   -- Num epochs before gamma changes
}

-- Argument parser
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

-- Timers
local data_tm = torch.Timer()
local tm = torch.Timer()
local epoch_tm = torch.Timer()
local graph_tm = torch.Timer()

-- Save opt
print('')
print('checkpoints will be saved to [ ' .. opt.outputdir .. ' ]')
print('')
torch.save(opt.outputdir .. '/' .. opt.name .. '_params', opt)

-- Random seeding
-- opt.manualSeed = torch.random(1, 10000)
print('Random seed: ' .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(opt.nThreads)
torch.setdefaulttensortype('torch.FloatTensor')

-- Initialize data
local DataLoader = paths.dofile('data/data.lua')
local data = DataLoader.new(opt.nThreads, opt)
print('Dataset train size: ', data:trainSize())

-- Set GPU and load net
-- Note: either start with VGGNet or resume previous model
cutorch.setDevice(opt.gpu)
local fc7_size = 4096
local output_size = 316
local detect_size = 80
local det_hidden_size = 4096
local vgg_net
local net
if opt.load_net_type == 'VGG' then
    -- Load net
    local loaded_net = torch.load(opt.load_net)

    -- Start building net
    local img_input = nn.Identity()()
    local c1 = cudnn.SpatialConvolution(3, 64, 3, 3, 1, 1, 1, 1)(img_input)
    c1.data.module.weight:copy(loaded_net:get(1).weight);
    c1.data.module.bias:copy(loaded_net:get(1).bias);
    local r2 = cudnn.ReLU()(c1)
    local c3 = cudnn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1)(r2)
    c3.data.module.weight:copy(loaded_net:get(3).weight);
    c3.data.module.bias:copy(loaded_net:get(3).bias);
    local r4 = cudnn.ReLU()(c3)
    local p5 = cudnn.SpatialMaxPooling(2, 2, 2, 2)(r4)
    local c6 = cudnn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1)(p5)
    c6.data.module.weight:copy(loaded_net:get(6).weight);
    c6.data.module.bias:copy(loaded_net:get(6).bias);
    local r7 = cudnn.ReLU()(c6)
    local c8 = cudnn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1)(r7)
    c8.data.module.weight:copy(loaded_net:get(8).weight);
    c8.data.module.bias:copy(loaded_net:get(8).bias);
    local r9 = cudnn.ReLU()(c8)
    local p10 = cudnn.SpatialMaxPooling(2, 2, 2, 2)(r9)
    local c11 = cudnn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1)(p10)
    c11.data.module.weight:copy(loaded_net:get(11).weight);
    c11.data.module.bias:copy(loaded_net:get(11).bias);
    local r12 = cudnn.ReLU()(c11)
    local c13 = cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)(r12)
    c13.data.module.weight:copy(loaded_net:get(13).weight);
    c13.data.module.bias:copy(loaded_net:get(13).bias);
    local r14 = cudnn.ReLU()(c13)
    local c15 = cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)(r14)
    c15.data.module.weight:copy(loaded_net:get(15).weight);
    c15.data.module.bias:copy(loaded_net:get(15).bias);
    local r16 = cudnn.ReLU()(c15)
    local p17 = cudnn.SpatialMaxPooling(2, 2, 2, 2)(r16)
    local c18 = cudnn.SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1)(p17)
    c18.data.module.weight:copy(loaded_net:get(18).weight);
    c18.data.module.bias:copy(loaded_net:get(18).bias);
    local r19 = cudnn.ReLU()(c18)
    local c20 = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)(r19)
    c20.data.module.weight:copy(loaded_net:get(20).weight);
    c20.data.module.bias:copy(loaded_net:get(20).bias);
    local r21 = cudnn.ReLU()(c20)
    local c22 = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)(r21)
    c22.data.module.weight:copy(loaded_net:get(22).weight);
    c22.data.module.bias:copy(loaded_net:get(22).bias);
    local r23 = cudnn.ReLU()(c22)
    local p24 = cudnn.SpatialMaxPooling(2, 2, 2, 2)(r23)
    local c25 = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)(p24)
    c25.data.module.weight:copy(loaded_net:get(25).weight);
    c25.data.module.bias:copy(loaded_net:get(25).bias);
    local r26 = cudnn.ReLU()(c25)
    local c27 = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)(r26)
    c27.data.module.weight:copy(loaded_net:get(27).weight);
    c27.data.module.bias:copy(loaded_net:get(27).bias);
    local r28 = cudnn.ReLU()(c27)
    local c29 = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)(r28)
    c29.data.module.weight:copy(loaded_net:get(29).weight);
    c29.data.module.bias:copy(loaded_net:get(29).bias);
    local r30 = cudnn.ReLU()(c29)
    local p31 = cudnn.SpatialMaxPooling(2, 2, 2, 2)(r30)
    local v32 = nn.View(-1):setNumInputDims(3)(p31)
    local f33 = nn.Linear(25088, 4096)(v32)
    f33.data.module.weight:copy(loaded_net:get(33).weight);
    f33.data.module.bias:copy(loaded_net:get(33).bias);
    local r34 = cudnn.ReLU()(f33)
    local d35 = nn.Dropout(0.5)(r34)
    local f36 = nn.Linear(4096, 4096)(d35)
    f36.data.module.weight:copy(loaded_net:get(36).weight);
    f36.data.module.bias:copy(loaded_net:get(36).bias);
    local r37 = cudnn.ReLU()(f36)
    vgg_net = nn.gModule({img_input}, {r37})

    -- Add layers
    local vgg_input = nn.Identity()()
    local detect_input = nn.Identity()()
    local fc7_plus = nn.JoinTable(1,1)({vgg_input, detect_input})
    local fc7_drop = nn.Dropout(0.5)(fc7_plus)
    local fc8 = nn.Linear(fc7_size + detect_size, output_size)(fc7_drop)
    fc8.data.module.weight:normal(0, 0.001);
    --fc8.data.module.weight:fill(0);
    fc8.data.module.bias:fill(-6.58);
    local out = cudnn.Sigmoid()(fc8)
    net = nn.gModule({vgg_input, detect_input}, {out})   
end
net:cuda()
net:training()
vgg_net:cuda()
vgg_net:training()
collectgarbage(); collectgarbage()
print('Loaded net')

-- Optimization
local parameters, gradParameters = net:getParameters()
local vgg_parameters, vgg_gradParameters = vgg_net:getParameters()

optim_state = {}
vgg_optim_state = {}

-- Criterion
local criterion = nn.MultiCriterion():add(nn.BCECriterion(), opt.lossweight)

-- Losses
local multiclass_loss = 0
local average_errors = 0
-- Load cuda data
--
local image_data = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize):zero()
local label = torch.Tensor(opt.batchSize, output_size)
local detect_annotation = torch.Tensor(opt.batchSize, detect_size)
image_data = image_data:cuda()
label = label:cuda()
detect_annotation = detect_annotation:cuda()
criterion:cuda()
print('GPU Data initialized')
collectgarbage(); collectgarbage()

-- Main evaluation function for classification net
function fevalclass(x)
    net:zeroGradParameters()

    local loss = 0

    -- Get data
    data_tm:reset(); data_tm:resume()
    local imdata, detectConf, detectHid, detectClass, detectionTable, lab, labinds = data:getBatch()
    detectConf = torch.reshape(detectConf, opt.batchSize, detect_size)
    data_tm:stop()

    image_data:copy(imdata)
    detect_annotation:copy(detectConf)
    
    label:copy(lab)
    local label_cpu = lab
    
    -- Forward through VGG net 
    local vgg_out = vgg_net:forward(image_data)

    -- Forward pass
    local output = net:forward({vgg_out, detect_annotation})

    -- Get error and gradient
    local out_cpu = torch.Tensor(output:size())
    multiclass_loss = criterion:forward(output, label)
    local df_do_c = criterion:backward(output, label)
   
    -- Get average num errors
    local rounded_out = torch.Tensor(output:size())
    rounded_out:copy(output)
    rounded_out[torch.gt(rounded_out, 0.5)] = 1
    rounded_out[torch.le(rounded_out, 0.5)] = -1
    local labelcopy = torch.Tensor(lab:size())
    labelcopy:copy(lab)
    labelcopy[torch.lt(labelcopy, 0.5)] = -1
    local mistakes = torch.cmul(labelcopy, rounded_out)
    mistakes[torch.gt(mistakes, 0)] = 0
    mistakes[torch.lt(mistakes, 0)] = 1
    local num_mistakes = mistakes:sum()
    average_errors = num_mistakes / opt.batchSize
    
    -- Backward pass   
    local dout = net:backward({vgg_out, detect_annotation}, df_do_c)

    loss = multiclass_loss

    return loss, gradParameters
end

function fevalvgg(x)
    --print('VGG grad: ' .. vgg_gradParameters:sum())
    vgg_net:zeroGradParameters()
 
    local loss = multiclass_loss
    
    -- Get input grad from net
    local dvgg_dnet = net.gradInput[1]
    vgg_net:backward(image_data, dvgg_dnet)

    return loss, vgg_gradParameters
end

-- Train
for epoch = opt.epoch_start, opt.num_epochs do
    epoch_tm:reset()

    -- Log results to files
    local lager = optim.Logger(paths.concat(opt.log, opt.name .. '_epoch' .. epoch .. '.log'))
    lager:setNames{'Multiclass loss', 'Average Errors', 'Importance loss', 'gradParams', 'gsnn_gradParams'}  

    -- Set learning rates
    local lr = opt.learning_rate * (opt.gamma ^ math.floor(epoch / opt.gamma_after))

    optim_config = {
        learningRate = lr,
        weightDecay = opt.weight_decay,
        momentum = opt.momentum,
    }
    vgg_optim_config = {
        learningRate = lr * opt.VGG_lr_weight,
        weightDecay = opt.weight_decay,
        momentum = opt.momentum,
    }

    for i = 1, math.min(data:trainSize(), opt.ntrain), opt.batchSize do
        tm:reset()

        -- Update classification network
        optim.sgd(fevalclass, parameters, optim_config, optim_state)
        
        -- Update VGG network
        optim.sgd(fevalvgg, vgg_parameters, vgg_optim_config, vgg_optim_state)     
                 
        print(('Epoch: [%d][%8d / %8d]\t Time: %.3f  DataTime: %.3f  '
               .. '  Multiloss: %.4f  Avg err: %.4f '
               .. '  Params: %.4f  GradParams: %.4f '
               .. '  VGGParams: %.4f  VGGGradParams: %.4f'):format(
               epoch, ((i-1)/opt.batchSize), 
               math.floor(math.min(data:trainSize(), opt.ntrain) / opt.batchSize),
               tm:time().real, data_tm:time().real,
               multiclass_loss, average_errors, 
               parameters:sum(), gradParameters:sum(), 
               vgg_parameters:sum(), vgg_gradParameters:sum()))
        print('Training: ' .. opt.name)
        lager:add{multiclass_loss, average_errors, importance_loss,
                  gradParameters:sum(), 0}  
        collectgarbage(); collectgarbage()
    end

    -- Save temporary net and/or saved nets
    collectgarbage(); collectgarbage()
    if epoch % opt.save_after == 0 then
        torch.save(opt.outputdir .. '/' .. opt.name .. '_multiclass_net_epoch_' .. epoch .. '.t7', net)
        torch.save(opt.outputdir .. '/' .. opt.name .. '_vgg_net_epoch_' .. epoch .. '.t7', vgg_net) 
    end
    collectgarbage(); collectgarbage()
    print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
           epoch, opt.num_epochs, epoch_tm:time().real))
end
