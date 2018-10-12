# Get most common objects, attributes and relationships from data
import json
import csv
import sys

# Counts to keep (don't matter here actually)
num_objs_keep = 200
num_atts_keep = 100
num_rel_keep = 25

# Data files
image_data_file = '/scratch/kdmarino/datasets/VisualGenome/visual_genome_python_driver/data/image_data.json'
object_data_file = '/scratch/kdmarino/datasets/VisualGenome/visual_genome_python_driver/data/objects.json'
att_data_file = '/scratch/kdmarino/datasets/VisualGenome/visual_genome_python_driver/data/attributes.json'
rel_data_file = '/scratch/kdmarino/datasets/VisualGenome/visual_genome_python_driver/data/relationships.json'
object_save_file = '/scratch/kdmarino/datasets/VisualGenome/visual_genome_python_driver/data/object_frequency.csv'
att_save_file = '/scratch/kdmarino/datasets/VisualGenome/visual_genome_python_driver/data/attribute_frequency.csv'
rel_save_file = '/scratch/kdmarino/datasets/VisualGenome/visual_genome_python_driver/data/relationship_frequency.csv'

# Load image data
img_data = json.load(open(image_data_file))

# Load object data
object_counts = {}
object_data = json.load(open(object_data_file))
for image_object_data in object_data:
	id = image_object_data['id']
	objects = image_object_data['objects']
	for obj in objects:
		obj_names = obj['names']
		for name in obj_names:
			name = name.replace(',', '') 
			name = name.replace('"', '')
                        name = name.replace("'", '')
			name = name.encode('ascii', 'backslashreplace')
                        name = name.strip()
			if name in object_counts.keys():
				object_counts[name] += 1
			else:
				object_counts[name] = 1

# Get top objects and save to csv file
sorted_obj_cats = sorted(object_counts, key=object_counts.get, reverse=True)
sorted_obj_counts = sorted(object_counts.values(), reverse=True)
fobj = open(object_save_file, 'wt')
obj_writer = csv.writer(fobj, delimiter=',', quoting=csv.QUOTE_NONE)
for i in range(len(sorted_obj_cats)):
	obj_writer.writerow((sorted_obj_cats[i], sorted_obj_counts[i]))

fobj.close()

# Get top attributes
att_counts = {}
att_data = json.load(open(att_data_file))
for image_att_data in att_data:
	id = image_att_data['id']
	attributes = image_att_data['attributes']
	for att in attributes:
		att_names = att['attributes']
		for name in att_names:
			name = name.replace(',', '') 
			name = name.replace('"', '')
                        name = name.replace("'", '')
			name = name.strip()
			name = name.encode('ascii', 'backslashreplace')
			if name in att_counts.keys():
				att_counts[name] += 1
			else:
				att_counts[name] = 1

# Get top attributes and save to csv file
sorted_att_cats = sorted(att_counts, key=att_counts.get, reverse=True)
sorted_att_counts = sorted(att_counts.values(), reverse=True)
fatt = open(att_save_file, 'wt')
att_writer = csv.writer(fatt, delimiter=',', quoting=csv.QUOTE_NONE)
for i in range(len(sorted_att_cats)):
	att_writer.writerow((sorted_att_cats[i], sorted_att_counts[i]))

fatt.close()

# Get top relationships
rel_counts = {}
rel_data = json.load(open(rel_data_file))
for image_rel_data in rel_data:
	id = image_rel_data['id']
	relationships = image_rel_data['relationships']
	for rel in relationships:
		name = rel['predicate']
		name = name.replace(',', '') 
		name = name.replace('"', '')
		name = name.replace("'", '')
		name = name.strip()
		name = name.encode('ascii', 'backslashreplace')
		if name in rel_counts.keys():
			rel_counts[name] += 1
		else:
			rel_counts[name] = 1

# Get top predicates and save to csv file
sorted_rel_cats = sorted(rel_counts, key=rel_counts.get, reverse=True)
sorted_rel_counts = sorted(rel_counts.values(), reverse=True)
frel = open(rel_save_file, 'wt')
rel_writer = csv.writer(frel, delimiter=',', quoting=csv.QUOTE_NONE)
for i in range(len(sorted_rel_cats)):
	rel_writer.writerow((sorted_rel_cats[i], sorted_rel_counts[i]))

frel.close()

print('Done!')

