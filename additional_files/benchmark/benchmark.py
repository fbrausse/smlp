import sys
sys.path.insert(1, '~/gitclones/Marabou/maraboupy')
sys.path.insert(2, '~/gitclones/Marabou')

from benchmark_functions import *
import inspect
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from maraboupy import Marabou
import os
import matplotlib.pyplot as plt

ACTIVIATION_FUNCTION = "relu"
n_samples = 100000
fun = ackley
batch_size = 64
epochs = 1000
input_dimensions = 2
plot_on = (input_dimensions == 1)

def create_data_set(b_function, ndim: int ,lb=0, ub=100):
    assert(ub > lb) 
    
    random_numbers = (ub - lb) * np.random.random((n_samples, ndim)) + lb
    training_size = int(0.8 * n_samples)
    testing_size = int(0.2 * n_samples)
    
    training_data = random_numbers[:training_size]
    testing_data = random_numbers[training_size:]
    training_labels = np.ones(shape=(training_size,1))
    testing_labels = np.ones(shape=(testing_size,1))

    for i in range(training_size):
        training_labels[i] = b_function(training_data[i])

    for i in range(testing_size):
        testing_labels[i] = b_function(testing_data[i])

    return training_data, training_labels, testing_data, testing_labels


def create_model(input_dim):
    model = keras.Sequential()
    model.add(keras.layers.Dense(units=32, activation=ACTIVIATION_FUNCTION, input_dim=input_dim)) 
    model.add(keras.layers.Dense(units=32, activation=ACTIVIATION_FUNCTION)) 
    model.add(keras.layers.Dense(units=32, activation=ACTIVIATION_FUNCTION)) 

    model.add(keras.layers.Dense(units=1))

    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=0.00001),  # Optimizer
        # Loss function to minimize
        loss=tf.keras.losses.MeanSquaredError(),
        # List of metrics to monitor
        metrics=[keras.metrics.RootMeanSquaredError()],
    )

    return model



if __name__ == "__main__":
    X_train, y_train, X_test, y_test = create_data_set(fun, input_dimensions, -10, 10)
    model = create_model(input_dimensions)
    model.fit(X_train,y_train, batch_size=batch_size, epochs=epochs)

    loss, metrics = model.evaluate(X_test, y_test)

    if plot_on:
        xs = np.linspace(-10,10,1000)
        plt.plot(xs,model.predict(xs))
        plt.show()

    tf.saved_model.save(model,'./my_model/')

    options = Marabou.createOptions(snc=True,numWorkers=4)
    network = Marabou.read_tf("./my_model/",modelType="savedModel_v2")
    inputVars = network.inputVars[0][0]
    outputVars = network.outputVars[0]

    network.setUpperBound(outputVars[0], -1.0)
    for m in range(len(inputVars)):
        network.setLowerBound(inputVars[m],-10.0)
        network.setUpperBound(inputVars[m],10.0)

    vals, stats = network.solve("marabou.log",options=options)

