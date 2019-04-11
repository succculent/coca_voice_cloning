import numpy as np

def normalizeAudio(labels):
	max_length = 0
	o_length = 0

	for i in range(0, len(labels)):
		#list of numpy arrays of variable size (audio)
		x, y = labels[i].shape
		if (y>max_length):
			max_length = y #double check if it is y or x lMAO

	new_labels = []

	for i in range(0, len(labels)):
		temp = np.zeros((x, max_length))
		x, y = labels[i].shape
		for j in range(0, x):
			for k in range(0, y):
				temp[j][k] = labels[i][j][k]
		new_labels.append(temp)

	return new_labels

a = np.ones((3, 3))
b = np.ones((3, 10))
c = np.ones((3, 7))
d = np.ones((3, 12))

l = [a, b, c, d]

for i in range(0, 4):
	print(l[i].shape)

print("done")

ll = normalizeAudio(l)

print("New stuff")

for i in range(0, 4):
	print(ll[i].shape)

print("done")