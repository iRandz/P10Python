import numpy as np
import pandas as pd

import Functions
import Settings
import SingleClassification
import FeatureDict as fD

settings = Settings.Settings()
# Prepare Data -------------------------------------------------------------
data = pd.read_csv("Data/CombinedSecondsLog.csv", sep=';')

print("Check for null:")
print(np.any(pd.isnull(data)))
print("---")

# Calculate 2nd order features
data_features: pd.DataFrame = Functions.calc_derived_features(data)

# Process data
data_features, data_labels = Functions.process_data(data_features, settings)

# Running average
runningRange = 50

data_features_running: pd.DataFrame = data_features.rolling(runningRange).sum()

dropList = range(0, runningRange)
print(dropList)
data_features_running = data_features_running.drop(dropList)
data_labels = data_labels.drop(dropList)

print(data_features_running)

data_features_running = Functions.normalize(data_features_running)

SingleClassification.single_classification(data_features_running, data_labels, settings)
