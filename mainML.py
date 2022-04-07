import pandas as pd
from enum import Enum

import DimensionalityCheck
import Functions
import SingleClassification


# Settings ----------------------------------------------------------------
class Settings:

    class ClassTarget(Enum):
        TYPE = 'Type'
        GENDER = 'Gender'

    dataFile = "V2Data.csv"
    target = ClassTarget.TYPE

    show2D = 0
    show3D = 1
    column1X = 0
    column2Y = 1
    column3Z = 2

    recalcManage = 0
    removeManage = 0
    removeOther = 1
    normalize = 1

    usePCA = 0
    useFeatSel = 1

    dimensionalityPCA = 4
    dimensionalitySel = 4

# Prepare Data -------------------------------------------------------------
data = pd.read_csv(Settings.dataFile, sep=';')
data_features, data_labels = Functions.ProcessData(data, Settings.recalcManage, Settings.removeManage, Settings.target)

# Preprocessing ----------------------------------------------------------
# Calculate 2nd order features
data_features = Functions.CalcDerivedFeatures(data_features)

# Normalize to 0-1 range
if Settings.normalize:
    data_features = Functions.Normalize(data_features)


# Test ideal dimensionality -------------------------------
DimensionalityCheck.DimensionalityCheck(data_features, data_labels)

# Single classification -----------------------------------
# SingleClassification.SingleClassification(data_features, data_labels, Settings)
