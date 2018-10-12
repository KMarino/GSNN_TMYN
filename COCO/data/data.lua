local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

local data = {}
local result = {}
local unpack = unpack and unpack or table.unpack

function data.new(n, opt_)
   opt_ = opt_ or {}
   local self = {}
   for k,v in pairs(data) do
      self[k] = v
   end

   local donkey_file = 'donkey_folder.lua'

   if n > 0 then
      local options = opt_
      self.threads = Threads(n,
                             function() require 'torch' end,
                             function(idx)
                                opt = options
                                tid = idx
                                local seed = (opt.manualSeed and opt.manualSeed or 0) + idx
                                torch.manualSeed(seed)
                                torch.setnumthreads(1)
                                print(string.format('Starting donkey with id: %d seed: %d', tid, seed))
                                assert(options, 'options not found')
                                assert(opt, 'opt not given')
                                print(opt)
                                paths.dofile(donkey_file)
                             end
      )
   else
      if donkey_file then paths.dofile(donkey_file) end
      self.threads = {}
      function self.threads:addjob(f1, f2) f2(f1()) end
      function self.threads:dojob() end
      function self.threads:synchronize() end
   end

   local nTrain = 0
   self.threads:addjob(function() return trainLoader:trainSize() end,
         function(c) nTrain = c end)
   self.threads:synchronize()
   self._trainSize = nTrain
   local nTest = 0
   self.threads:addjob(function() return trainLoader:testSize() end,
         function(c) nTest = c end)
   self.threads:synchronize()
   self._testSize = nTest

   -- Use opt_ to decide which threads to push here
   for i = 1, n do
      self.threads:addjob(self._getFromThreads, self._pushResult)
   end

   return self
end

function data._getFromThreads()
    assert(opt.batchSize, 'opt.batchSize not found')
    local getDetection
    if opt.runmode == 'baseline' then
        getDetection = 0
    else
        getDetection = 1
    end

    return trainLoader:sample(opt.batchSize, getDetection, opt.detectMode)
end

-- Important note - you can only have one kind of thread running or bad things happen
function data._pushResult(...)
    local res = {...}
    if res == nil then
        self.threads:synchronize()
    end
    result[1] = res
end

function data:getBatch()
   -- queue another job
   self.threads:addjob(self._getFromThreads, self._pushResult)
   self.threads:dojob()
   --self.threads:synchronize()
   local res = result[1]
   result[1] = nil
   if torch.type(res) == 'table' then
      return unpack(res)
   end
   print(type(res))
   return res
end

-- Test batch
function data:getTestBatch(start_idx)
   assert(opt.batchSize, 'opt.batchSize not found')
   local getDetection
   if opt.runmode == 'baseline' then
       getDetection = 0
   else
       getDetection = 1
   end
   return trainLoader:sampleTest(opt.batchSize, getDetection, opt.detectMode, start_idx)
end

function data:trainSize()
    return self._trainSize
end
function data:testSize()
    return self._testSize
end

return data
