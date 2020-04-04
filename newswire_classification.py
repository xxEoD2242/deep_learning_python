from vector_utils import vectorize_sequences, to_one_hot
from plotting_utils import plot_metric
from keras.datasets import reuters
from keras import models, layers

import numpy as np

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)

# Partial train to evluate performance
x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
# The output here is a 46-diemnsional vector, which each dimension contains
# a vector with a probability that the input was that class. All of the dimensions
# sum to one.
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
			  loss='categorical_crossentropy',
			  metrics=['accuracy'])

model.fit(partial_x_train,
					partial_y_train,
					epochs=9,
					batch_size=512,
					validation_data=(x_val, y_val))

results = model.evaluate(x_test, one_hot_test_labels)

print(results)

predictions = model.predict(x_test)

print(predictions[0].shape)
print(np.sum(predictions[0]))
print(np.argmax(predictions[0]))
"""

history = model.fit(partial_x_train,
					partial_y_train,
					epochs=20,
					batch_size=512,
					validation_data=(x_val, y_val))

print(history.history)
loss = history.history['loss']
val_loss = history.history['val_loss']

plot_metric(loss, val_loss, 'Loss')

accuracy = history.history['acc']
val_acc = history.history['val_acc']

plot_metric(accuracy, val_acc, 'Accuracy')

"""