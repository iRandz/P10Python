import numpy as np
import pandas as pd

import DimensionalityCheck
import Functions
import OvRROC
import Settings
import SingleClassification
import SingleRegression

settings = Settings.Settings()
# Prepare Data -------------------------------------------------------------
file = "data_lowPlayTime.csv"
data = pd.read_csv(file, sep=';')

settings.dataFile = file

print("Check for null:")
print(np.any(pd.isnull(data)))
print("---")

# Calculate 2nd order features
data_features = Functions.calc_derived_features(data)

# Process data
data_features, data_labels = Functions.process_data(data_features, settings)

# Standardize
if settings.normalize:
    data_features = Functions.normalize(data_features)

if settings.test == settings.CurrentTest.DIMENSIONALITY:
    # Test ideal dimensionality -------------------------------
    DimensionalityCheck.dimensionality_check(data_features, data_labels, settings)
elif settings.test == settings.CurrentTest.CLASSIFICATION:
    # Single classification -----------------------------------
    SingleClassification.single_classification(data_features, data_labels, settings)
elif settings.test == settings.CurrentTest.REGRESSION:
    SingleRegression.single_classification(data_features, data_labels, settings)
elif settings.test == settings.CurrentTest.ROC:
    OvRROC.calcOvR(data_features, data_labels, settings)
else:
    # Do nothing
    print("What u doing?")

