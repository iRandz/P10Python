import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import csv
import pandas as pd

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

# Prepare Data -------------------------------------------------------------

data = pd.read_csv("Tracking_log.csv", sep=';')

data_features = data.copy()
data_labels = data_features.pop('Type')

data_features = np.array(data_features)

#print(data_labels)
#print("----------")
#print(data_features)

# Prepare and train model -------------------------------------------------------

normalize = layers.Normalization()
normalize.adapt(data_features)

data_model = tf.keras.Sequential([
    normalize,
    layers.Dense(64),
    layers.Dense(3)
])

data_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                   optimizer=tf.optimizers.Adam(),
                   metrics=['accuracy'])

data_model.fit(data_features, data_labels, epochs=10)

# Validate model -------------------------------------------------------------------------

print("--------")
print("Single value check:")

probability_model = tf.keras.Sequential([data_model, tf.keras.layers.Softmax()])

dataTest = pd.read_csv("Tracking_test.csv", sep=';')

dataTest_features = data.copy()
dataTest_labels = dataTest_features.pop('Type')

dataTest_features = np.array(data_features)
predictions = probability_model.predict(dataTest_features)

print(predictions[0])
print(dataTest_labels[0])

print("--------")
print("Validation:")

print(data_model.evaluate(dataTest_features,  dataTest_labels, verbose=2))

data_model.save("C:\\Users\\chrbj\\PycharmProjects\\P10Python\\Model")

