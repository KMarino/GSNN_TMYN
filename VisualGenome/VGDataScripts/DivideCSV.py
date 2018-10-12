# Divide up the csv files so they are small enough for torch to handle
require 'json'
savedir = 'data/manyjson/'

# Object data
object_data_file = 'data/objects.json'
object_data = json.load(open(object_data_file))

count = 1
for data in object_data:
	save_file = savedir + 'object_' + str(count) + '.json'
	f = open(save_file, 'w')
        json.dump(data, f)
	f.close()
	count = count + 1

# Attribute data
attribute_data_file = 'data/attributes.json'
att_data = json.load(open(attribute_data_file))

count = 1
for data in att_data:
	save_file = savedir + 'attribute_' + str(count) + '.json'
	f = open(save_file, 'w')
	json.dump(data, f)
	f.close()
	count = count + 1

# Relationship data
relationship_data_file = 'data/relationsships.json'
rel_data = json.load(open(relationship_data_file))

count = 1
for data in rel_data:
	save_file = savedir + 'relationship_' + str(count) + '.json'
	f = open(save_file, 'w')
	json.dump(data, f)
	f.close()
	count = count + 1


