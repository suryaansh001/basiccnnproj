# # import tensorflow as tf
# # import keras 
# # import numpy as   np 
# # import pandas as pd
# import tensorflow as tf
# import numpy as np
# import cv2
# from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
# from keras.models import Sequential



# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# train=(x_train,y_train)

# # x_train,y_train,x_test,y_test = x_train/255.0,y_train,x_test/255.0,y_test#making all the vbalues between 0 and 1 
# # x_test, y_test = x_test / 255.0, y_test / 255.0
# def input_fn(train,training=True):
#     dataset=tf.data.Dataset.from_tensor_slices((x_train,y_train))
#     if training:
#         dataset=dataset.shuffle(1000)
#     return dataset


# #applying convolutoiion layer 
# model=tf.keras.Sequential([
#     tf.keras.Conv2D(32,(3,3),input_shape=(28,28,1)),# it shows the the kernel ot the filter applied is of 3*3 and the number of filters are 32
#     tf.keras.MaxPooling2D(2,2),
#     tf.keras.Conv2D(64,(3,3)),# it shows the the kernel ot the filter applied is of 3*3 and the number of filters are 64
#     tf.keras.MaxPooling2D(2,2),
#     tf.keras.Conv2D(64,(3,3)),# it shows the the kernel ot the filter applied is of 3*3 and the number of filters are 64
#     tf.keras.MaxPooling2D(2,2),
#     tf.keras.Flatten(),
#     tf.keras.Dense(64,activation='relu'),# 64 is the one layer the ANN part starts from here 
#     tf.keras.Dense(10,activation='softmax')# it shows the number of neurons in each layer and activation function is softmax
# ])

# model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# model.fit(input_fn(train,training=True),epochs=5)

# model.evaluate(input_fn(train,training=False))


# print(model.summary())
# print(model.evaluate(input_fn(train,training=False)))



# print("Model trained")


import tensorflow as tf
import numpy as np

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize the pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Reshape the data to add the channel dimension as the model is trained on 28*28 
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Define the input function for creating the dataset
def input_fn(data, labels, training=True):
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    if training:
        dataset = dataset.shuffle(1000).batch(32)#batch 32 means that it will take 32 images train and then shuffle all tehn take 32 images in this manner it allows it to change the weights some more than 1000 times
    else:
        dataset = dataset.batch(32)
    return dataset

# Create the model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)), # Conv2D layer with 32 filters and a kernel size of 3x3
    tf.keras.layers.MaxPooling2D(2, 2), # MaxPooling layer with a pool size of 2x2
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'), # Conv2D layer with 64 filters and a kernel size of 3x3
    tf.keras.layers.MaxPooling2D(2, 2), # MaxPooling layer with a pool size of 2x2
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'), # Conv2D layer with 64 filters and a kernel size of 3x3
    tf.keras.layers.MaxPooling2D(2, 2), # MaxPooling layer with a pool size of 2x2
    #starting ANN 
    tf.keras.layers.Flatten(), # Flatten layer to convert the 2D matrix to a 1D vector
    tf.keras.layers.Dense(64, activation='relu'), # Dense layer with 64 neurons and ReLU activation
    tf.keras.layers.Dense(10, activation='softmax') # Dense layer with 10 neurons and softmax activation
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
epochs=5
model.fit(input_fn(x_train, y_train, training=True), epochs=epochs)#epochs 

# Evaluate the model
loss, accuracy = model.evaluate(input_fn(x_test, y_test, training=False))
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# # Print the model summary
# print(model.summary())\
model.save("mnsist.h5")

