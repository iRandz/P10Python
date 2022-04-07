import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from enum import Enum

import Functions
import Classification


class ClassTarget(Enum):
    TYPE = 'Type'
    GENDER = 'Gender'


# Settings ----------------------------------------------------------------
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
data = pd.read_csv(dataFile, sep=';')
data_features, data_labels = Functions.ProcessData(data, recalcManage, removeManage, target)

# Preprocessing ----------------------------------------------------------
# Calculate 2nd order features
data_features = Functions.CalcDerivedFeatures(data_features)

# Normalize to 0-1 range
if normalize:
    data_features = Functions.Normalize(data_features)

# Feature selection / PCA
data_features = Functions.FeatureSelection(data_features, data_labels, usePCA, useFeatSel, dimensionalitySel, dimensionalityPCA)

print(data_features.columns)

# Force wrongly labelled data into 'Other'
#data_labels, data = Functions.HandleInvalidData(data_labels, removeOther, data)

# Classificaiton -------------------------------------------------------------------------
score = Classification.Classify(data_features, data_labels)

# Display plots ---------------------------------------------------------------------------
if target == ClassTarget.TYPE:
    mydict = {'Journey': 'green',
              'Manage': 'blue',
              'Assault': 'red',
              'Other': 'black'}
elif target == ClassTarget.GENDER:
    mydict = {'Male': 'blue',
              'Female': 'red',
              'Other': 'black'}
else:
    sys.exit("Unknown target. Can't show plots." + target.value)

if show3D:
    fig = plt.figure(1, figsize=(8, 6))
    ax = fig.add_subplot(projection='3d')

    ax.scatter(data_features.iloc[:, column1X], data_features.iloc[:, column2Y], data_features.iloc[:, column3Z],
               c=data_labels.map(mydict), edgecolor="k")
    if not usePCA:
        ax.set_xlabel(data_features.columns[column1X])
        ax.set_ylabel(data_features.columns[column2Y])
        ax.set_zlabel(data_features.columns[column3Z])

    fake_handles = [mpatches.Patch(color=item) for item in mydict.values()]
    label = mydict.keys()
    ax.legend(fake_handles, label, loc='upper right', prop={'size': 10})

    ax.text(0, 0, 0, ("%.2f" % score).lstrip("0"))

    plt.show()

if show2D:

    plt.scatter(data_features.iloc[:, column1X], data_features.iloc[:, column2Y], c=data_labels.map(mydict))
    if not usePCA:
        plt.xlabel(data_features.columns[column1X])
        plt.ylabel(data_features.columns[column2Y])

    fake_handles = [mpatches.Patch(color=item) for item in mydict.values()]
    label = mydict.keys()
    plt.legend(fake_handles, label, loc='upper right', prop={'size': 10})

    plt.show()
