# Utilities for plotting parameters of the NN

import matplotlib.pyplot as plt

def plot_metric(metric, val_metric, metric_name):
	# We should always clear the plot if there was one
	plt.clf()

	epochs = range(1, len(metric)+1)
	plt.plot(epochs, metric, 'bo', label=('Training ' + metric_name))
	plt.plot(epochs, val_metric, 'b', label=('Validation ' + metric_name))
	plt.title('Training and Validation ' + metric_name)
	plt.xlabel('Epochs')
	plt.ylabel(metric_name.capitalize())
	plt.legend()

	plt.show()