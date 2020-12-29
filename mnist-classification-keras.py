'''
Queen's University, Fall 2020
Course: COGS 400 - Neural and Genetic Computing
Assignment 2 - Backpropogation Part B - Implementing Keras
Student Number: 20062694

Sources:
https://www.tensorflow.org/tutorials/quickstart/beginner
'''
from sklearn.metrics import confusion_matrix
import numpy as np
import tensorflow as tf

def getData():
    #dataset is built in
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    #int to float
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return x_train, y_train, x_test, y_test

def buildModel():
    #using keras
    #input is the image 28x28
    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
    ])
    return model

def predict(x_train, y_train, model):
    #x_train is the unlabelled data
    predictions = model(x_train[:1]).numpy()
    #softmax activation to get probabilities
    tf.nn.softmax(predictions).numpy()
    #calculate loss
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss_fn(y_train[:1], predictions).numpy()
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='sparse_categorical_accuracy', dtype=None
)
    accuracy(y_train[:1], predictions).numpy()
    print(str(accuracy))

    return loss_fn, predictions

def main():
    x_train, y_train, x_test, y_test = getData()
    model = buildModel()
    loss_fn, predictions = predict(x_train,y_train, model)
    #using adam -adaptive moment estimation- for best results
    model.compile(optimizer='adam',loss=loss_fn, metrics= 'accuracy')
    #finetune params and train
    model.fit(x_train, y_train, epochs=20)
    #test model
    model.evaluate(x_test,  y_test, verbose=2)
    probability_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])
    #5 epochs
    probability_model(x_test[:1])

if __name__ == '__main__':
    main()
