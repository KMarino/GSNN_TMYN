-- Dataset loader for VisualGenome Test

require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')
local ffi = require 'ffi'
local class = require('pl.class')
local dir = require 'pl.dir'
local tablex = require 'pl.tablex'
local argcheck = require 'argcheck'
require 'sys'
require 'xlua'
require 'image'
require 'string'
require 'csvigo'

local dataset = torch.class('dataLoader')

local initcheck = argcheck{
    pack=true,
    help=[[
      A dataset class for images in a flat folder structure (folder-name is class-name).
      Optimized for extremely large datasets (upwards of 14 million images).
      Tested only on Linux (as it uses command-line linux utilities to scale up)
    ]],
    {name="datadir",
     type="string",
     help="root data directory"},
   
    {name="sampleSize",
     type="table",
     help="a consistent sample size to resize the images"},

    {name="split",
     type="number",
     help="Percentage of split to go to Training"
    },

    {name="samplingMode",
     type="string",
     help="Sampling mode: random | balanced ",
     default = "balanced"},

    {name="verbose",
     type="boolean",
     help="Verbose mode during initialization",
     default = false},

    {name="percat",
     type="number",
     help="1-shot num ex per cat",
     default=0},

    {name="loadSize",
     type="table",
     help="a size to load the images to, initially",
     opt = true},

    {name="sampleHookTrain",
     type="function",
     help="applied to sample during training(ex: for lighting jitter). "
        .. "It takes the image path as input",
     opt = true},

    {name="sampleHookTest",
     type="function",
     help="applied to sample during testing",
     opt = true},
}

function dataset:__init(...)

    -- argcheck
    local args =  initcheck(...)
    print(args)
    for k,v in pairs(args) do self[k] = v end

    if not self.loadSize then self.loadSize = self.sampleSize; end

    if not self.sampleHookTrain then self.sampleHookTrain = self.defaultSampleHook end
    if not self.sampleHookTest then self.sampleHookTest = self.defaultSampleHook end

    -- Load data
    self.metadata = json.load(self.datadir .. 'visual_genome_python_driver/data/image_data.json')
    self.splits = torch.load(self.datadir .. '/torchdata/splits_minival.t7')
    self.numSamples = #self.metadata

    if self.verbose then print(self.numSamples ..  ' samples found.') end
  
    -- Get 1-shot dataset
    local oneshot_data = torch.load(self.datadir .. '/torchdata/dataset_100cats_minivalsplit.t7')
    self.oneshot_cats = oneshot_data[1]
    self.oneshot_catdict = oneshot_data[2]
    self.oneshot_train_1x = oneshot_data[3]
    self.oneshot_train_2x = oneshot_data[4]
    self.oneshot_train_5x = oneshot_data[5]
    self.oneshot_test = oneshot_data[6]
    self.oneshot_val = oneshot_data[7]
    self.oneshot_negatives = torch.load(self.datadir .. '/torchdata/negative_examples_1shot.t7')

    -- Set splits
    if self.split == 100 then
        self.ListTrain = self.splits.perm
        self.ListTest = {}
    elseif self.split == 90 then
        self.ListTrain = self.splits.split90Train
        self.ListTest = self.splits.split90Test
    elseif self.split == 80 then
        self.ListTrain = self.splits.split80Train
        self.ListTest = self.splits.split80Test
    elseif self.split == 75 then
        self.ListTrain = self.splits.split75Train
        self.ListTest = self.splits.split75Test
    elseif self.split == 50 then
        self.ListTrain = self.splits.split50Train
        self.ListTest = self.splits.split50Test
    end

    -- Get imagenet mean
    local img_mean_dir = self.datadir .. '/torchdata/ilsvrc_2012_mean.t7'
    self.img_mean = torch.load(img_mean_dir).img_mean:transpose(3,1)
end

-- get size of data
function dataset:size()
    return self.numSamples
end

-- Number of training examples
function dataset:trainSize()
    return #self.ListTrain
end

-- Return number of training examples for one-shot
function dataset:trainSizeOneShot()
    if self.percat == 0 then
        return 0
    elseif self.percat == 1 then
        return #self.oneshot_train_1x
    elseif self.percat == 2 then
        return #self.oneshot_train_2x
    elseif self.percat == 5 then 
        return #self.oneshot_train_5x
    end
end

-- Return size of test set for one-shot
function dataset:testSizeOneShot()
    return #self.oneshot_test
end

-- Number of testing examples
function dataset:testSize()
    return #self.ListTest
end

function dataset:getDetections(image_id)
    -- Check number of detections
    assert(#detections <= 80, "Shouldn't have more detections than classes")

    return detections
end

-- getTrainExample
function dataset:getTrainExample(getDetection)
    -- Get index
    local randind = math.ceil(torch.uniform() * #self.ListTrain)
    local index = self.ListTrain[randind]
    local image_id = self.metadata[index].image_id
 
    -- Load torch data
    local torchdata = torch.load(self.datadir .. '/torchdata/data_' .. image_id .. '.t7')
    local imgpath = self.datadir .. '/' .. torchdata.name
    local label = torchdata.present
    local label_inds = torch.Tensor(label:size()):zero()
    for i = 1, label:size(1) do
        if label[i] == 1 then
            label_inds[i] = i
        else
            label_inds[i] = 0
        end
    end
     
    -- Load image
    local img = self:sampleHookTrain(imgpath, self.img_mean)

    if img == nil then
        print('image is nil!')
        print('id is ' .. image_id)
    end

    -- Load detections (if getDetection is 1)
    if getDetection == 1 then
        local detections = torchdata.detections
        return img, detections, label, label_inds
    else
        return img, label, label_inds
    end
end

-- getTrainExampleLowdata
-- Same as get train, but specify num train examples to use
function dataset:getTrainExampleLowdata(getDetection, numTrain)
    -- Get index
    assert(numTrain <= #self.ListTrain)
    local randind = math.ceil(torch.uniform() * numTrain)
    local index = self.ListTrain[randind]
    local image_id = self.metadata[index].image_id
 
    -- Load torch data
    local torchdata = torch.load(self.datadir .. '/torchdata/data_' .. image_id .. '.t7')
    local imgpath = self.datadir .. '/' .. torchdata.name
    local label = torchdata.present
    local label_inds = torch.Tensor(label:size()):zero()
    for i = 1, label:size(1) do
        if label[i] == 1 then
            label_inds[i] = i
        else
            label_inds[i] = 0
        end
    end
     
    -- Load image
    local img = self:sampleHookTrain(imgpath, self.img_mean)

    if img == nil then
        print('image is nil!')
        print('id is ' .. image_id)
    end

    -- Load detections (if getDetection is 1)
    if getDetection == 1 then
        local detections = torchdata.detections
        return img, detections, label, label_inds
    else
        return img, label, label_inds
    end
end

-- Same as get train, but uses one-shot images and loads different labels
function dataset:getTrainExampleOneShot(getDetection, percat)
    -- Get index from datasets list
    local index
    if percat == 1 then
        local randind = math.ceil(torch.uniform() * #self.oneshot_train_1x) 
        index = self.oneshot_train_1x[randind]   
    elseif percat == 2 then
        local randind = math.ceil(torch.uniform() * #self.oneshot_train_2x)
        index = self.oneshot_train_2x[randind]
    elseif percat == 5 then
        local randind = math.ceil(torch.uniform() * #self.oneshot_train_5x)
        index = self.oneshot_train_5x[randind]
    else
        error('Invalid percat option')
    end
    local image_id = self.metadata[index].image_id

    -- Load torch data
    local torchdata = torch.load(self.datadir .. '/torchdata/data_' .. image_id .. '.t7')
    local imgpath = self.datadir .. '/' .. torchdata.name
    local label_old = torchdata.present
    local label = torchdata.newcat_present
    assert(label, 'Cant find label for new categories')
     
    -- Load image
    local img = self:sampleHookTrain(imgpath, self.img_mean)

    if img == nil then
        print('image is nil!')
        print('id is ' .. image_id)
    end

    -- Load detections (if getDetection is 1)
    if getDetection == 1 then
        local detections = torchdata.detections
        return img, detections, label, label_old
    else
        return img, label, label_old
    end
end

-- Get negative training examples for 1-shot
function dataset:getTrainExampleOneShotNeg(getDetection)
    -- Get index from datasets list
    local randind = math.ceil(torch.uniform() * #self.oneshot_negatives)
    local index = self.oneshot_negatives[randind]   
    local image_id = self.metadata[index].image_id

    -- Load torch data
    local torchdata = torch.load(self.datadir .. '/torchdata/data_' .. image_id .. '.t7')
    local imgpath = self.datadir .. '/' .. torchdata.name
    local label_old = torchdata.present
    local label = torch.Tensor(100):zero() -- Hack, hardcoded 100 in here
     
    -- Load image
    local img = self:sampleHookTrain(imgpath, self.img_mean)
    if img == nil then
        print('image is nil!')
        print('id is ' .. image_id)
    end

    -- Load detections (if getDetection is 1)
    if getDetection == 1 then
        local detections = torchdata.detections
        return img, detections, label, label_old
    else
        return img, label, label_old
    end
end

-- getTestExample
function dataset:getTestExample(getDetection, test_idx)
    -- Get index
    local index = self.ListTest[test_idx]
    local image_id = self.metadata[index].image_id
    --print(image_id)

    -- Get torch data
    local torchdata = torch.load(self.datadir .. '/torchdata/data_' .. image_id .. '.t7')
    local imgpath = self.datadir .. '/' .. torchdata.name
    local label = torchdata.present
    local label_inds = torch.Tensor(label:size()):zero()
    for i = 1, label:size(1) do
        if label[i] == 1 then
            label_inds[i] = i
        else
            label_inds[i] = 0
        end
    end   
 
    -- Get image
    --local img = self:sampleHookTrain(imgpath, self.img_mean)
    local img = self:sampleHookTest(imgpath, self.img_mean) 

    -- Load detections (if getDetection is 1)
    if getDetection == 1 then
        local detections = torchdata.detections
        --local detections = self:getDetections(image_id)
        --print(#detections)
        return img, detections, label, label_inds
    else
        return img, label, label_inds
    end
end

-- getTestExampleOneShot
function dataset:getTestExampleOneShot(getDetection, test_idx)
    -- Get index
    local index = self.oneshot_test[test_idx]
    local image_id = self.metadata[index].image_id

    -- Get torch data
    local torchdata = torch.load(self.datadir .. '/torchdata/data_' .. image_id .. '.t7')
    local imgpath = self.datadir .. '/' .. torchdata.name
    local label = torchdata.newcat_present
    local label_old = torchdata.present
 
    -- Get image
    --local img = self:sampleHookTrain(imgpath, self.img_mean)
    local img = self:sampleHookTest(imgpath, self.img_mean)    

    -- Load detections (if getDetection is 1)
    if getDetection == 1 then
        local detections = torchdata.detections
        return img, detections, label, label_old
    else
        return img, label, label_old
    end
end

-- Converts a table of samples (and corresponding labels) to a clean tensor
local function tableToOutput(self, dataTable, labelTable, labelindsTable)
    local quantity = #dataTable
    assert(dataTable[1]:dim() == 3)
    local data = torch.Tensor(quantity, self.sampleSize[1], self.sampleSize[2], self.sampleSize[3]):zero()
    local labels = torch.Tensor(quantity, labelTable[1]:size(1)):zero()
    local labelinds = torch.Tensor(quantity, labelindsTable[1]:size(1)):zero()
    for i = 1, quantity do
        data[i]:copy(dataTable[i])
        labels[i]:copy(labelTable[i])
        labelinds[i]:copy(labelindsTable[i])
    end

    return data, labels, labelinds
end

-- Converts a table of detections and sorts them into right form for annotation net
local function convertDetectionData(self, detectionTable) 
    local quantity = #detectionTable
    local num_det_class = 80 
    local hidden_size = 4096   
    local detectConf = torch.Tensor(quantity*num_det_class, 1):zero()
    local detectHid = torch.Tensor(quantity*num_det_class, hidden_size):zero()   
    local detectClass = torch.Tensor(quantity*num_det_class, num_det_class):zero()
    for i = 1, quantity do
        local detections = detectionTable[i]
        --print(detections)
        for j = 1, #detections do
            -- Get detection data
            local detection = detections[j]
            local class_ind = detection.class_ind
            local conf = detection.conf
            local hidden = detection.hidden
             
            -- Put into correct place in data
            detectConf[(i-1)*num_det_class + class_ind] = conf
            detectHid[(i-1)*num_det_class + class_ind] = hidden
            detectClass[(i-1)*num_det_class + class_ind][class_ind] = 1
        end
    end

    return detectConf, detectHid, detectClass
end

-- sampler, samples from the training set.
function dataset:sample(quantity, getDetection)
    assert(quantity)
    local dataTable = {}
    local labelTable = {}
    local labelindsTable = {}
    local detectionTable = {}
    for i=1,quantity do
        local img, detections, label, label_inds 
        if getDetection == 1 then
            img, detections, label, label_inds = self:getTrainExample(getDetection)
            table.insert(detectionTable, detections)
        else
            img, label, label_inds = self:getTrainExample(getDetection)
        end
        table.insert(dataTable, img)
        table.insert(labelTable, label)
        table.insert(labelindsTable, label_inds)
    end
    local data, labels, labelinds = tableToOutput(self, dataTable, labelTable, labelindsTable)   
 
    if getDetection == 1 then
        local detectConf, detectHid, detectClass = convertDetectionData(self, detectionTable) 
        return data, detectConf, detectHid, detectClass, detectionTable, labels, labelinds
    else
        return data, labels, labelinds
    end
end

-- Same as other sampler, but with now data
function dataset:sampleLowdata(quantity, getDetection, numTrain)
    assert(quantity)
    local dataTable = {}
    local labelTable = {}
    local labelindsTable = {}
    local detectionTable = {}
    for i=1,quantity do
        local img, detections, label, label_inds 
        if getDetection == 1 then
            img, detections, label, label_inds = self:getTrainExampleLowdata(getDetection, numTrain)
            table.insert(detectionTable, detections)
        else
            img, label, label_inds = self:getTrainExampleLowdata(getDetection, numTrain)
        end
        table.insert(dataTable, img)
        table.insert(labelTable, label)
        table.insert(labelindsTable, label_inds)
    end
    local data, labels, labelinds = tableToOutput(self, dataTable, labelTable, labelindsTable)   
 
    if getDetection == 1 then
        local detectConf, detectHid, detectClass = convertDetectionData(self, detectionTable) 
        return data, detectConf, detectHid, detectClass, detectionTable, labels, labelinds
    else
        return data, labels, labelinds
    end
end

-- Sampler for onesot
function dataset:sampleOneShot(quantity, getDetection, percat, negex)
    assert(quantity)
    assert(getDetection)
    assert(percat)
    assert(negex)
    local dataTable = {}
    local labelTable = {}
    local labeloldTable = {}
    local detectionTable = {}
    for i=1,quantity do
        local img, detections, label, label_inds 
        if getDetection == 1 then
            img, detections, label, label_old = self:getTrainExampleOneShot(getDetection, percat)
            table.insert(detectionTable, detections)
        else
            img, label, label_old = self:getTrainExampleOneShot(getDetection, percat)
        end
        table.insert(dataTable, img)
        table.insert(labelTable, label)
        table.insert(labeloldTable, label_old)
    end
    local data, labels, labelold = tableToOutput(self, dataTable, labelTable, labeloldTable)    
 
    if negex == 1 then
        local dataTable_neg = {}
        local labelTable_neg = {}
        local labeloldTable_neg = {}
        local detectionTable_neg = {}
        for i = 1, quantity do
            local img, detections, label, label_inds
            if getDetection == 1 then
                img, detections, label, label_old = self:getTrainExampleOneShotNeg(getDetection)
                table.insert(detectionTable_neg, detections)
            else
                img, label, label_old = self:getTrainExampleOneShotNeg(getDetection)
            end
            table.insert(dataTable_neg, img)
            table.insert(labelTable_neg, label)
            table.insert(labeloldTable_neg, label_old)
        end
        
        local data_neg, labels_neg, labelold_neg = tableToOutput(self, dataTable_neg, labelTable_neg, labeloldTable_neg) 
        if getDetection == 1 then
            local detectConf, detectHid, detectClass = convertDetectionData(self, detectionTable)
            local detectConf_neg, detectHid_neg, detectClass_neg = convertDetectionData(self, detectionTable_neg)
            return data, data_neg, detectConf, detectConf_neg, detectHid, detectHid_neg, detectClass, detectClass_neg, labels, lables_neg, labelold, labelold_neg
        else
            return data, data_neg, labels, labels_neg, labelold, labelold_neg 
        end
    else
        if getDetection == 1 then
            local detectConf, detectHid, detectClass = convertDetectionData(self, detectionTable)
            return data, detectConf, detectHid, detectClass, detectionTable, labels, labelold
        else
            return data, labels, labelold
        end
    end
end

-- sampler, samples from test set.
function dataset:sampleTest(quantity, getDetection, start_idx)
    assert(quantity)
    
    -- Check that we haven't run out of examples
    if start_idx + quantity - 1 > #self.ListTest then
        return nil
    end

    local dataTable = {}
    local labelTable = {}
    local labelindsTable = {}
    local detectionTable = {}
    for i=1,quantity do
        local img, detections, label, label_inds 
        if getDetection == 1 then
            img, detections, label, label_inds = self:getTestExample(getDetection, start_idx+i-1)
            table.insert(detectionTable, detections)
        else
            img, label, label_inds = self:getTestExample(getDetection, start_idx+i-1)
        end
        table.insert(dataTable, img)
        table.insert(labelTable, label)
        table.insert(labelindsTable, label_inds)
    end
    local data, labels, labelinds = tableToOutput(self, dataTable, labelTable, labelindsTable)
    
    if getDetection == 1 then
        local detectConf, detectHid, detectClass = convertDetectionData(self, detectionTable) 
        return data, detectConf, detectHid, detectClass, detectionTable, labels, labelinds
    else
        return data, labels, labelinds
    end
end

-- sampler, samples from test set of one shot
function dataset:sampleTestOneShot(quantity, getDetection, start_idx)
    assert(quantity)
    
    -- Check that we haven't run out of examples
    if start_idx + quantity - 1 > #self.oneshot_test then
        return nil
    end

    local dataTable = {}
    local labelTable = {}
    local labeloldTable = {}
    local detectionTable = {}
    for i=1,quantity do
        local img, detections, label, label_inds 
        if getDetection == 1 then
            img, detections, label, label_old = self:getTestExampleOneShot(getDetection, start_idx+i-1)
            --print(detections)
            table.insert(detectionTable, detections)
        else
            img, label, label_old = self:getTestExampleOneShot(getDetection, start_idx+i-1)
        end
        table.insert(dataTable, img)
        table.insert(labelTable, label)
        table.insert(labeloldTable, label_old)
    end
    local data, labels, labelold = tableToOutput(self, dataTable, labelTable, labeloldTable)
    
    if getDetection == 1 then
        local detectConf, detectHid, detectClass = convertDetectionData(self, detectionTable) 
        --print(detectConf)
        return data, detectConf, detectHid, detectClass, detectionTable, labels, labelold
    else
        return data, labels, labelold
    end
end
return dataset
