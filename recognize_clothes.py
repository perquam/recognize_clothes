import tensorflow as tf
import plot_image as pi
#API to add some functionality
from tensorflow import keras

data = keras.datasets.fashion_mnist  
# Split data into training and test parts
(train_images, train_labels), (test_images, test_labels) = data.load_data()

# Transform information about greyscale from range [0, 255] to [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0

# Show one of the images.
pi.plt.imshow(train_images[7], cmap=pi.plt.cm.binary)
#plt.show()

# Build all the layers.
model = keras.Sequential([
	keras.layers.Flatten(input_shape=(28, 28)), # [[]] -> []
	# each neuron represents a pixel.
	keras.layers.Dense(128, activation="relu"),  # Next layer,
	# fully connected layer. Activation is "arbitrary".
	keras.layers.Dense(10, activation="softmax") # Last layer.
	# Also sum of individual values of activation of neurons
	# adds up to 1.
	])

# There are different optimizers,
# different  loss functions, these needs to be checked up.
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Epoch decides how many times the neural network
# sees the same images. Increasing it does not necessarily
# increase accuracy.
model.fit(train_images, train_labels, epochs=5)

# To see how it works, this needs to be tested
# with test part of the data.
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print("Tested acc: ", test_acc)

predictions = model.predict(test_images)
# Choose the number of object you want to check.
i=10
pi.plt.figure(figsize=(6,3))
pi.plt.subplot(1,2,1)
pi.plot_image(i, predictions[i], test_labels, test_images)
pi.plt.show()