import pandas as pd
from enum import Enum

import DimensionalityCheck
import Functions
import Regression
import SingleClassification


# Settings ----------------------------------------------------------------
import SingleRegression


class Settings:
    class ClassTarget(Enum):
        TYPE = 'Type'
        GENDER = 'Gender'

    class RegressionTarget(Enum):
        PLAYTIME = 'Weekly playtime'
        AGE = 'Age'
        SPAM = 'Shots fired'
        KILLS = 'Kills'
        RESOURCES = 'Resources'
        LORE = 'Lore interactions'

    class CurrentTest(Enum):
        DIMENSIONALITY = 0
        CLASSIFICATION = 1
        REGRESSION = 2

    dataFile = "V2Data.csv"
    test = CurrentTest.REGRESSION
    classifier_target = ClassTarget.TYPE
    regressor_target = RegressionTarget.RESOURCES

    show2D = 1
    show3D = 0
    column1X = 0
    column2Y = 1
    column3Z = 2

    recalcManage = 0
    removeManage = 0
    removeOther = 1
    normalize = 0

    usePCA = 0
    useFeatSel = 1

    dimensionalityPCA = 4
    dimensionalitySel = 2


# Prepare Data -------------------------------------------------------------
data = pd.read_csv(Settings.dataFile, sep=';')

# Calculate 2nd order features
data_features = Functions.calc_derived_features(data)

# Process data
data_features, data_labels = Functions.process_data(data_features, Settings)

# Normalize to 0-1 range
if Settings.normalize:
    data_features = Functions.normalize(data_features)

if Settings.test == Settings.CurrentTest.DIMENSIONALITY:
    # Test ideal dimensionality -------------------------------
    DimensionalityCheck.dimensionality_check(data_features, data_labels)
elif Settings.test == Settings.CurrentTest.CLASSIFICATION:
    # Single classification -----------------------------------
    SingleClassification.single_classification(data_features, data_labels, Settings)
elif Settings.test == Settings.CurrentTest.REGRESSION:
    SingleRegression.single_classification(data_features, data_labels, Settings)
else:
    # Do nothing
    print("What u doing?")
