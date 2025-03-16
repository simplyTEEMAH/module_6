python --version
python -m pip show tensorflow
import tensorflow as tf
import keras
import numpy as np
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
(x_train, y_train), (x_test,y_test)= fashion_mnist.load_data() #load the dataset
# prepare and preprocess the data
# Normalize pixel values to be between 0 and 1
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
x_train, x_test = x_train / 255.0, x_test / 255.0
# # reshape the data to correctly process as 2D images with a single channel
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
# check the shape of the data
print(x_train.shape)
print(x_test.shape)
# create a sequential model
model=Sequential()
# Add convolutional, pooling, dropout and dense layers
model.add (Conv2D(32, kernel_size=(3,3), activation = "relu", input_shape = (28,28,1)))
model.add (Conv2D(32, kernel_size=(3,3), activation = "relu"))
model.add (Conv2D(64, kernel_size=(3,3), activation = "relu"))
model.add (Conv2D(64, kernel_size=(3,3), activation = "relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add (Conv2D(64, kernel_size=(3,3), activation = "relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add (Conv2D(128, kernel_size=(3,3), activation = "relu"))
model.add (Dropout (0.25)) # regularisation technique to prevent overfitting
model.add (Flatten()) # reshape the 3D output to a 1D output
model.add (Dropout(0.5)) # add dropout layer with a dropout rate of 50%
model.add (Dense(10, activation ="softmax"))
model.summary()
# compile the model
model.compile(loss=keras.losses.sparse_categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
# train the model
model.fit (x_train, y_train, batch_size=128, epochs=10,validation_data= (x_test, y_test))
'''
The model was trained from an initial 63.74% accuracy and ended with a 91.67% accuracy
It has a validation accuracy of 91.41% 
This implies that the model is generalising well to unseen data and is not overfitting
'''
plt.plot(model.history.history['accuracy'], label='accuracy')
plt.plot(model.history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
'''
The model has a test accuracy of 91.41%
This implies that the model is generalising well and can make accurate predictions on unseen data.
'''
# make predictions of at least two images (select two random indices from the test set)
random_indices = np.random.choice(len(x_test), 2, replace=False)
# select two random images and its labels
images = x_test[random_indices]
labels = y_test[random_indices]
# make predictions for the randomly selected images
predictions = model.predict(images)
for i, image in enumerate (images):
    plt.subplot (1,2, i+1)
    plt.imshow (image, cmap=plt.cm.binary)
    plt.title (f"Predicted: {class_names[np.argmax (predictions[i])]}")
    plt.axis("off")
plt.show()
'''
The model predicts a shirt and a pullover
'''

