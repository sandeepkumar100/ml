# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 14:27:54 2018

This script is an example for training a CNN model using keras, the script accepts
epochs, learning_rate and batch as argument.
Default values:
    epochs: 5
    learning_rate: 0.001
    batch: 32
How use this script:

python keras_CIFAR_10_ML_demo.py --epochs <ep> --learning_rate <lr> --batch <batch_size>

@author: jaydeep.deka
"""
import argparse
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
#from matplotlib import pyplot


def get_arguments():
    """
    Description: This function can be used to get arguments from the user that
    are planning in the text box.
    Args: None
    Returns: Argument dictionary
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch', type=str, default=32, help='No of images in each batch.')
    return parser.parse_args()

def preProcessData():
    """
    Description: This function will load data for us using the available API for our example
    Preprocess the data normalisation
    Args: None
    Returns: Processed datasets
    """
    # Step1: Loading the data
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    # Step2: Normalise the data
    # normalize inputs from 0-255 to 0.0-1.0
    X_train = X_train.astype('float32')/255.0
    X_test = X_test.astype('float32')/ 255.0

    # Step3: One hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    
    return X_train, X_test, y_train, y_test

def createModel(num_classes):
    """
    Description: This function is used to create a CNN model.
    Args: num_classes
    Returns: model
    """
    # Create the model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def train(X_train, y_train, X_test, y_test, model, batch_size, epochs, lr):
    """
    Description: This function can be used to train the model
    Args: X_train, y_train, X_test, y_test, model, batch_size, epochs, lr
    Returns: model
    """
    decay = lr/epochs

    # Define the optimiser
    sgd = SGD(lr=lr, momentum=0.9, decay=decay, nesterov=False)

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    print(model.summary())

    # Fit the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)
    return model

def saveTheCheckpoint(model):
    """
    Description: This function can be used to train the model
    Args: model
    Returns: None
    """
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")

def main():
    args = get_arguments()
    epochs = args.epochs
    lr = args.learning_rate
    batch_size = args.batch

    print("Received data from the user learning_rate={}, epochs={}, batch_size={}".format(lr,epochs,batch_size))
    # Load and preprocess the data
    X_train, X_test, y_train, y_test = preProcessData()
    
    # Optional: Show the images in 3x3 grid
    # create a grid of 3x3 images
#    for i in range(0, 9):
#    	pyplot.subplot(330 + 1 + i)
#    	pyplot.imshow(toimage(X_train[i]))
#    # show the plot
#    pyplot.show()
    
    # Create the model
    model = createModel(num_classes=y_test.shape[1])
    
    # Train the model
    model = train(X_train, y_train, X_test, y_test, model, batch_size, epochs, lr)
    
    # Save the model
    saveTheCheckpoint(model)
    
if __name__=='__main__':
    main()
