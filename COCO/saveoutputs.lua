-- Plot the precision recall curves and save to file
require 'torch'
require 'optim'
require 'csvigo'

opt = {
    load_file = '',
    save_file = '',
}

-- Argument parser
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

local testdata = torch.load(opt.load_file)
local outputs = testdata[18]
local output_table = {}
for i = 1, #outputs do
    output_batch = outputs[i]
    assert(torch.type(output_batch) == 'torch.FloatTensor')
    for j = 1, output_batch:size(1) do
        output_ex_table = {}
        for k = 1, output_batch:size(2) do
            table.insert(output_ex_table, output_batch[j][k])        
        end
        table.insert(output_table, output_ex_table)
    end    
end

-- Output table should be size of testing set
assert(#output_table == #outputs*outputs[1]:size(1))
-- Output dimension should be number of classes
assert(#output_table[1] == outputs[1]:size(2))

-- Save outputs
csvigo.save(opt.save_file, output_table)

