require 'torch'
require 'paths'

-- Get all the file names
local paths1 = paths.dir('VG_100K/')
local paths2 = paths.dir('VG_100K_2/')

-- For each files, save to text file
file = io.open('VG_files.txt', 'w')
io.output(file)
for i = 1, #paths1 do
    if (paths1[i] ~= '.') and (paths1[i] ~= '..') then
        io.write('VG_100K/' .. paths1[i] .. '\n')
    end
end
for i = 1, #paths2 do
    if (paths2[i] ~= '.') and (paths2[i] ~= '..') then
        io.write('VG_100K_2/' .. paths2[i] .. '\n')
    end
end
io.close()
