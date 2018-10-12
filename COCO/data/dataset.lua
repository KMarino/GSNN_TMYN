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
--require 'hdf5'

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

    {name="imagedir",
     type="string",
     help="image root data directory"},
   
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
    local args = initcheck(...)
    print(args)
    for k,v in pairs(args) do self[k] = v end

    if not self.loadSize then self.loadSize = self.sampleSize; end

    if not self.sampleHookTrain then self.sampleHookTrain = self.defaultSampleHook end
    if not self.sampleHookTest then self.sampleHookTest = self.defaultSampleHook end

    self.traindata = torch.load(self.datadir .. '/data/coco_train_fromhdf5.t7')
    self.numTrain = self.traindata.imageIds:size(1)
    self.minivaldata = torch.load(self.datadir .. '/data/minival2014data.t7')
    self.numTest = #self.minivaldata

    if self.verbose then print((self.numTrain + self.numTest) ..  ' samples found.') end
  
    -- Get imagenet mean
    local img_mean_dir = self.datadir .. '/torchdata/ilsvrc_2012_mean.t7'
    self.img_mean = torch.load(img_mean_dir).img_mean:transpose(3,1)
end

-- Number of training examples
function dataset:trainSize()
    return self.numTrain
end

-- Number of testing examples
function dataset:testSize()
    return self.numTest
end

-- getTrainExample
function dataset:getTrainExample(getDetection, detectmode)
    -- Get index
    local randind = math.ceil(torch.uniform() * self.numTrain)
    local image_id = self.traindata.imageIds[randind]
 
    -- Load torch data
    local torchdata = torch.load(self.datadir .. '/torchdata/train/data_' .. image_id .. '.t7')
    local imgpath = string.format('%s/train2014/COCO_train2014_%012d.jpg', self.imagedir, image_id)

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
        local detections
        if detectmode == 'voc' then
            detections = torchdata.voc_detections
        elseif detectmode == 'coco' then
            detections = torchdata.coco_detections
        else
            error('Invalid detectmode option')
        end
        assert(img); assert(detections); assert(label); assert(label_inds)
        return img, detections, label, label_inds
    else
        assert(img); assert(label); assert(label_inds)
        return img, label, label_inds
    end
end

-- getTestExample
function dataset:getTestExample(getDetection, detectmode, test_idx)
    -- Get index
    local image_id = self.minivaldata[test_idx].id

    -- Load torch data
    local torchdata = torch.load(self.datadir .. '/torchdata/val/data_' .. image_id .. '.t7')
    local imgpath = string.format('%s/val2014/COCO_val2014_%012d.jpg', self.imagedir, image_id)
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
    local img = self:sampleHookTest(imgpath, self.img_mean)
    
    -- Load detections (if getDetection is 1)
    if getDetection == 1 then
        local detections
        if detectmode == 'voc' then
            detections = torchdata.voc_detections
        elseif detectmode == 'coco' then
            detections = torchdata.coco_detections
        else
            error('Invalid detectmode option')
        end
        assert(img); assert(detections); assert(label); assert(label_inds)
        return img, detections, label, label_inds
    else
        assert(img); assert(label); assert(label_inds)
        return img, label, label_inds
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
local function convertDetectionData(self, detectionTable, detectMode) 
    local quantity = #detectionTable
    
    -- Check detect mode
    local num_det_class
    if detectMode == 'voc' then
        num_det_class = 20
    elseif detectMode == 'coco' then
        num_det_class = 80
    else
        error('Invalid detectMode option..')
    end
    assert(num_det_class)
    local hidden_size = 4096  
    local detectConf = torch.Tensor(quantity*num_det_class, 1):zero()
    local detectHid = torch.Tensor(quantity*num_det_class, hidden_size):zero()   
    local detectClass = torch.Tensor(quantity*num_det_class, num_det_class):zero()
    for i = 1, quantity do
        local detections = detectionTable[i]
        assert(detections)
        --print(detections)
        for j = 1, #detections do
            -- Get detection data
            local detection = detections[j]
            local class_ind = detection.class_ind
            local conf = detection.conf
        
            -- Put into correct place in data
            detectConf[(i-1)*num_det_class + class_ind] = conf
            detectClass[(i-1)*num_det_class + class_ind][class_ind] = 1
        end
    end

    return detectConf, detectHid, detectClass
end

-- sampler, samples from the training set.
function dataset:sample(quantity, getDetection, detectMode)
    assert(quantity)
    assert(getDetection)
    assert(detectMode)
    local dataTable = {}
    local labelTable = {}
    local labelindsTable = {}
    local detectionTable = {}
    for i=1,quantity do
        local img, detections, label, label_inds 
        if getDetection == 1 then
            img, detections, label, label_inds = self:getTrainExample(getDetection, detectMode)
            table.insert(detectionTable, detections)
        else
            img, label, label_inds = self:getTrainExample(getDetection, detectMode)
        end
        table.insert(dataTable, img)
        table.insert(labelTable, label)
        table.insert(labelindsTable, label_inds)
    end
    local data, labels, labelinds = tableToOutput(self, dataTable, labelTable, labelindsTable)   
 
    if getDetection == 1 then
        local detectConf, detectHid, detectClass = convertDetectionData(self, detectionTable, detectMode) 
        return data, detectConf, detectHid, detectClass, detectionTable, labels, labelinds
    else
        return data, labels, labelinds
    end
end

-- sampler, samples from test set.
function dataset:sampleTest(quantity, getDetection, detectMode, start_idx)
    assert(quantity)
    assert(getDetection)
    assert(detectMode)
    assert(start_idx)   
 
    -- Check that we haven't run out of examples
    if start_idx + quantity - 1 > self.numTest then
        return nil
    end

    local dataTable = {}
    local labelTable = {}
    local labelindsTable = {}
    local detectionTable = {}
    for i=1,quantity do
        local img, detections, label, label_inds 
        if getDetection == 1 then
            img, detections, label, label_inds = self:getTestExample(getDetection, detectMode, start_idx+i-1)
            table.insert(detectionTable, detections)
        else
            img, label, label_inds = self:getTestExample(getDetection, detectMode, start_idx+i-1)
        end
        table.insert(dataTable, img)
        table.insert(labelTable, label)
        table.insert(labelindsTable, label_inds)
    end
    local data, labels, labelinds = tableToOutput(self, dataTable, labelTable, labelindsTable)
    
    if getDetection == 1 then
        local detectConf, detectHid, detectClass = convertDetectionData(self, detectionTable, detectMode) 
        return data, detectConf, detectHid, detectClass, detectionTable, labels, labelinds
    else
        return data, labels, labelinds
    end
end

return dataset
