-- File to load in images
require 'image'
paths.dofile('dataset.lua')
require 'json'

local loadSize   = {3, opt.loadSize}
local sampleSize = {3, opt.fineSize}

local function loadImage(path)
    local input = image.load(path, 3, 'float')
    -- find the smaller dimension, and resize it to loadSize[2] (while keeping aspect ratio)
    local iW = input:size(3)
    local iH = input:size(2)
    if iW < iH then
        input = image.scale(input, loadSize[2], loadSize[2] * iH / iW)
    else
        input = image.scale(input, loadSize[2] * iW / iH, loadSize[2])
    end
    return input
end

-- channel-wise mean and std. Calculate or load them from disk later in the script.
local mean,std
--------------------------------------------------------------------------------
-- Hooks that are used for each image that is loaded

-- function to load the image, jitter it appropriately (random crops etc.)
local trainHook = function(self, path, img_mean)
    collectgarbage()
    local input = loadImage(path)
    local iW = input:size(3)
    local iH = input:size(2)

    -- do random crop
    local oW = sampleSize[2]
    local oH = sampleSize[2]
    local h1 = math.ceil(torch.uniform(1e-2, iH-oH))
    local w1 = math.ceil(torch.uniform(1e-2, iW-oW))
    local out = image.crop(input, w1, h1, w1 + oW, h1 + oH)
    assert(out:size(2) == oW)
    assert(out:size(3) == oH)
    -- do hflip with probability 0.5
    if torch.uniform() > 0.5 then out = image.hflip(out); end
    
    -- Do imagent normalization
    out = out*255
    out = out:index(1, torch.LongTensor{3,2,1})
    out = out - image.scale(img_mean, sampleSize[2], sampleSize[2], 'bilinear'):float()
    return out
end

-- function to load the image. Does not do jitter or flips
local testHook = function(self, path, img_mean)
    collectgarbage()
    local input = loadImage(path)
    out = image.scale(input, sampleSize[2], sampleSize[2], 'bilinear')
    local oW = sampleSize[2]
    local oH = sampleSize[2]
    assert(out:size(2) == oW)
    assert(out:size(3) == oH)
    
    -- Do imagent normalization
    out = out*255
    out = out:index(1, torch.LongTensor{3,2,1})
    out = out - image.scale(img_mean, sampleSize[2], sampleSize[2], 'bilinear'):float()
    return out
end

--------------------------------------
-- trainLoader

-- Get meta data
if not opt.percat then
    opt.percat = 0
end
trainLoader = dataLoader{
    loadSize = {3, loadSize[2], loadSize[2]},
    sampleSize = {3, sampleSize[2], sampleSize[2]},
    split = 80,
    datadir = opt.dataloc,
    imagedir = opt.imageloc,
    verbose = true,
    percat = opt.percat,
}

trainLoader.sampleHookTrain = trainHook
trainLoader.sampleHookTest = testHook
collectgarbage()
