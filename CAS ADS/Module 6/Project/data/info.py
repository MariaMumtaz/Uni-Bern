import csv

import numpy as np


#1. read the dataset
x = []
y = []
with open('spambase.data', 'rt') as csvfile:
	data = csv.reader(csvfile, delimiter=',')
	for raw in data:
		x.append(np.array(raw[:-1], dtype=np.float32))
		y.append(int(raw[-1]))
		
x = np.array(x)
y = np.array(y)


n_ch = x.shape[-1]

#normalize inputs
for ch in range(n_ch):
	ch_v = x[:, ch]
	norm = ch_v.max()
	x[:, ch] /= norm

print (x.shape, y.shape)
	
# x - 57 parameters descibing exch mail
# y == 1: spam, 0 - not spam.


#2. build a fully connected classifier network that will classify the mails.
#3. check which parameters affect decision the most (biggest gradient wrt to answer)
#4. build a classifier that doesn't use 1, 2, 10 most significant parameters. Does it still perform better then random?