from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, MaxPool2D
from interface import MnistClassifierInterface
import numpy as np

'''
Define a CNN model for MNIST classification using Keras Sequential API
the model consists of several convolutional layers (Conv2D), max pooling layers (MaxPool2D),
and dropout layers for regularization, followed by fully connected layers (Dense).
the output layer uses 'softmax' activation for multi-class classification for digits 0-9.

Conv2D - extracts features using filters.
MaxPool2D - reduces the dimensionality of the data and highlights important features.
Dropout - prevents overfitting.
Flatten - converts data into a one-dimensional format for fully connected layers.
Dense - performs classification, turning extracted features into class labels.
'''

class CNNModel(MnistClassifierInterface):
    def __init__(self):
        self.model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
        Conv2D(32, kernel_size=(3, 3), activation='relu' ),
        MaxPool2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        Conv2D(64, kernel_size=(3, 3), activation='relu' ),
        MaxPool2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same' ),
        Conv2D(128, kernel_size=(3, 3), activation='relu' ),
        MaxPool2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(10, activation = "softmax") 
        ])
        
        # Compiling model with optimizer and loss reduction
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    def train(self, X_train, y_train):
        X_train = X_train.reshape(-1, 28, 28, 1)
        self.model.fit(X_train, y_train, epochs=25, batch_size=128)
    
    def predict(self, X_test):
        X_test = X_test.reshape(-1, 28, 28, 1)
        predictions = self.model.predict(X_test)
        
        # Returns the index of the maximum value along the row
        return np.argmax(predictions, axis=1)
