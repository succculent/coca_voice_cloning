import numpy as np

def normalizeAudio(labels):
	max_length = 0
	o_length = 0

	for i in range(0, len(labels)-1):
		#list of numpy arrays of variable size (audio)
		x, y = labels[i].size()
		if (y>max_length):
			max_length = y #double check if it is y or x lMAO

	new_labels = []

	for i in range(0, len(labels)-1):
		temp = np.zeros(o_length, max_length)
		x, y = labels[i].size()
		for j in range(0, x-1):
			for k in range(0, y-1):
				temp[j][k] = labels[i][j][k]
		new_labels.append(temp)