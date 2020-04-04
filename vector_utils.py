# Methods for working with tensors

import numpy as np

# Given a sequence of integers, create a tensor with the
# given dimension, thereby making the data structure the
# same length for all entries.

# One-Hot encoding of input vectors. Take the sample review
# and encode a 1 for each word indicie, making every vector
# of length 10,000.
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

def to_one_hot(labels, dimension=46):
	results = np.zeros((len(labels), dimension))
	for i, label in enumerate(labels):
		results[i, label] = 1
	return results