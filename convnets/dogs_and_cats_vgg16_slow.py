from keras import layers, models, optimizers
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from plotting_utils import plot_metric
import os

current_path = os.getcwd()

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

base_dir = os.path.join(current_path, 'dogs_vs_cats_small')

train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
validation_dir = os.path.join(base_dir, 'validation')

conv_base = VGG16(weights='imagenet',
				  include_top=False,
				  input_shape=(150,150,3))

conv_base.trainable = False

# Generator the proper image sizes
# Add data augmentation to increase the number of samples seen
# by the network to better develop abstractions.
train_datagen = ImageDataGenerator(rescale=1./255,
								   rotation_range=40,
								   width_shift_range=0.2,
								   height_shift_range=0.2,
								   shear_range=0.2,
								   zoom_range=0.2,
								   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
	train_dir,
	target_size=(150,150),
	batch_size=20,
	class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
	validation_dir,
	target_size=(150,150),
	batch_size=20,
	class_mode='binary')

# Build the model

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
# model.add(layers.Dropout(0.5)) # added with data augmentation
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
			  optimizer=optimizers.RMSprop(lr=2e-5),
			  metrics=['acc'])

history = model.fit_generator(
	train_generator,
	steps_per_epoch=100,
	epochs=100,
	validation_data=validation_generator,
	validation_steps=50)

model.save('cats_and_dogs_small_vgg16_slow.h5')

acc = history.history["acc"]
val_acc = history.history["val_acc"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

plot_metric(acc, val_acc, 'Accuracy')
plot_metric(loss, val_loss, 'Loss')