# Importing TensorFlow and tf.keras
import tensorflow as tf

# Importing helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# Importing and loading fashion data set used to train model
fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Setting class names to sort dataset
class_names = ['T-shirt/top', 'Pants', 'Hoodie', 'Dress', 'Jacket', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Display how many images and at what dimensions
train_images.shape

# How many images
len(train_labels)

train_labels

test_images.shape

len(test_labels)

# Preprocess the data (pixel values land between 0-255)
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# Scale these values between 0-1, then feed them into neural network model (divide each value by 255)
train_images = train_images / 255.0
test_images = test_images / 255.0

# Here we check to see that the data is ion the correct format so we are ready to build and train the neural network (display first 25 images)
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# Build the model here

# Set up layers (First building block of a neural network)
# Deep learning mostly consists of chaining together simple layers. 
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(10)
    ])

"""This first layer flatten is used to transform the format of the images from a 2D array (28x28) into a one-dimensional array (28*28) 784 pixels. 
Upstacking rows of pixles in the image and lining them up

After they are flattened the network consists of two dense layers. These are neural layers. 
First one has 128 nodes (neurons)
The second returns a logits array with a length of 10. 

Each node contains a score that indications the current image blongs to one of the 10 classes."""

# Compile the model

"""
Loss function:  this measures how accurate the model is during training
Optimizer:  This is how the model is updated based on the data it sees and the loss function
Metrics:  Monitors the training and testing steps. 
"""

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# Train the model

"""
1. Feed the training data to the model (train_images and train_labels)
2. The model learns to associate images and labels
3. Ask the model to make predictions about a test set (test_images array)
4. Verify that the predictions match the labels from the test labels array
"""

# Feed the model
model.fit(train_images, train_labels, epochs=10)

# Evaluate the accuracy
test_loss,  test_acc = model.evaluate(test_images, test_labels, verbose=2)

print('\nTest Accuracy: ', test_acc)

# Make predictions
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_images)

# first prediction
predictions[0]

# label of first
print(np.argmax(predictions[0]))

test_labels[0]

# Graph to look at full set of 10 class predictions
def plot_image(i, predictions_array, true_label, img):
    true_label, img=true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label], 100*np.max(predictions_array), class_names[true_label]), color=color)

def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0,1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# Verify predictions
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i], test_labels)
plt.show()

i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i], test_labels)
plt.show()

# plot several images with their predictions, model can be wrong when confident
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

# Using the trained model

# collect image from dataset, see if I can enter other data

# visualize 50
i = 50
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i], test_labels)
plt.show()

img = test_images[50]
print(img.shape)

# add image to a batch as tf.keras is optimized for a collection of examples
img = (np.expand_dims(img,0))
print(img.shape)

# predict correct label for image
predictions_single = probability_model.predict(img)
print(predictions_single)

plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()

np.argmax(predictions_single[0])