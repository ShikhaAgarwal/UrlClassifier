import csv, itertools
import numpy as np
import re
from random import shuffle

labels_given = ['Arts', 'Adult', 'Business', 'Computers', 'Games', 'Health', 'Home', 'Kids', 'News', 'Recreation', 'Reference', 'Science', 'Shopping', 'Society', 'Sports']
# path = '/Users/shikha/UMass/summer2018/url_classifier/datasets/'

def find_dataset_size(path):
	label_size = {}
	for label in labels_given:
		filename = path + label + '.csv'
		with open(filename, 'r') as f_in:
			reader = csv.reader(f_in, delimiter = ",")
			label_size[label] = len(list(reader))

	return label_size

def calculate_other_label_length(label_size, label, sorted_label_size):
	length_required = label_size[label] / (len(label_size) - 1)
	index = 0
	diff = 0
	for k, v in sorted_label_size:
		if v < length_required:
			index += 1
			diff += length_required - v
			continue
		break
	length_greater = length_required + (diff / len(label_size) - index)

	return index, length_greater

def create_dataset(path, label):
	print "Loading Dataset..."	
	
	filename = path + label + '.csv'
	label_size = find_dataset_size(path)

	XY = []
	with open(filename, 'r') as f_in:
		next(f_in)
		column23 = [ cols[1:3] for cols in csv.reader(f_in, delimiter=",") ]
	XY = XY + column23

	sorted_label_size = sorted(label_size.iteritems(), key=lambda (k, v): (v,k))
	index, length_greater = calculate_other_label_length(label_size, label, sorted_label_size)

	i = 0
	for k, v in sorted_label_size:
		if k == label:
			continue
		label_count = 0
		if i >= index:
			label_count = length_greater
		else:
			label_count = v

		filename = path + k + '.csv'
		with open(filename, 'r') as f_in:
			next(f_in)
			column23 = []
			for cols in itertools.islice(csv.reader(f_in, delimiter=","), label_count):
				column23.append(cols[1:3])
		i += 1
		XY = XY + column23

	shuffle(XY)
	Y=[]
	for col in XY:
		if col[1] == label:
			Y.append(1)
		else:
			Y.append(0)
	X = [col[0] for col in XY]
	# print Y[100:115], X[100:115]
	X = np.array(X)
	Y = np.array(Y)

	return X, Y

# print find_dataset_size(path)
# create_dataset(path, 'Arts')