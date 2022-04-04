import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sklearn.feature_selection
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import RFECV
from enum import Enum

import Functions


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
column3Z = 3

recalcManage = 0
removeManage = 0
removeOther = 1
normalize = 1

usePCA = 0
useFeatSel = 1
useNewFeatSel = 0

dimensionalityPCA = 4
dimensionalitySel = 4

neighbors = 5  # KNN
nnHiddenLayers = (10, 15, 6)  # NN
ovrEstimator = KNeighborsClassifier(n_neighbors=neighbors)  # OvR
svmKernel = 'linear'  # SVM

# Prepare Data -------------------------------------------------------------
data = pd.read_csv(dataFile, sep=';')
data_features = Functions.ProcessData(data, recalcManage, removeManage)

data_labels = data_features.pop(target.value)

simplify = False
if simplify:
    for x in range(data_labels.size):
        if data_labels[x] == 'Assault' or data_labels[x] == 'Journey':
            data_labels[x] = 'Other'


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
data_labels, data = Functions.HandleInvalidData(data_labels, removeOther, data)

# Create test/train split
X_train, X_test, y_train, y_test = train_test_split(
        data_features, data_labels, test_size=0.2, random_state=42
    )

# ML -----------------------------------------------------------------------------------------
# Prepare and train model
neigh = KNeighborsClassifier(n_neighbors=neighbors)
neigh.fit(X_train, y_train)

neural = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=nnHiddenLayers, random_state=1, max_iter=1000)
neural.fit(X_train, y_train)

ovr = OneVsRestClassifier(ovrEstimator)
ovr.fit(X_train, y_train)

svmCLF = svm.SVC(kernel=svmKernel)
svmX_train = X_train
svmX_test = X_test
# Most model / split feature selection
if useNewFeatSel:
    selector = RFECV(svmCLF)
    selector = selector.fit(X_train, y_train)
    svmX_train = selector.transform(X_train)
    svmX_test = selector.transform(X_test)

    print(selector.get_feature_names_out())
    print(selector.ranking_)
    print(selector.n_features_)
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (accuracy)")
    plt.plot(
        range(1, len(selector.grid_scores_) + 1),
        selector.grid_scores_,
    )
    plt.show()

svmCLF.fit(svmX_train, y_train)

# Validate model
print("\n -----------------------------")
print("K nearest neighbor")
# print(neigh.predict(X_test))
# print(np.array(y_test))
print(neigh.score(X_test, y_test))
y_true = y_test
y_pred = neigh.predict(X_test)
print(balanced_accuracy_score(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))

print("\n -----------------------------")
print("Multi layer perceptron")
print("Iterations: " + str(neural.n_iter_))
print("Layers: " + str(neural.hidden_layer_sizes))
print(neural.score(X_test, y_test))
y_pred = neural.predict(X_test)
print(balanced_accuracy_score(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))

print("\n -----------------------------")
print("One versus rest")
print("Using: " + str(ovr.estimator))
print(ovr.score(X_test, y_test))
y_pred = ovr.predict(X_test)
print(balanced_accuracy_score(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))

print("\n -----------------------------")
print("SVM")
print("Kernel: " + str(svmCLF.kernel))
print(svmCLF.score(svmX_test, y_test))
y_pred = svmCLF.predict(svmX_test)
print(balanced_accuracy_score(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))

score = neigh.score(X_test, y_test)

# Display plots ---------------------------------------------------------------------------
mydict = {'Journey': 'green',
          'Manage': 'blue',
          'Assault': 'red',
          'Other': 'black'}

if show3D:
    fig = plt.figure(1, figsize=(8, 6))
    ax = fig.add_subplot(projection='3d')

    if usePCA:
        ax.scatter(pca_features[:, column1X], pca_features[:, column2Y], pca_features[:, column3Z],
                   c=data_labels.map(mydict), edgecolor="k")
    else:
        ax.scatter(data_features.iloc[:, column1X], data_features.iloc[:, column2Y], data_features.iloc[:, column3Z],
               c=data_labels.map(mydict), edgecolor="k")
        ax.set_xlabel(data_features.columns[column1X])
        ax.set_ylabel(data_features.columns[column2Y])
        ax.set_zlabel(data_features.columns[column3Z])

    fake_handles = [mpatches.Patch(color=item) for item in mydict.values()]
    label = mydict.keys()
    ax.legend(fake_handles, label, loc='upper right', prop={'size': 10})

    ax.text(0, 0, 0, ("%.2f" % score).lstrip("0"))

    plt.show()

if show2D:
    if usePCA:
        plt.scatter(pca_features[:, column1X], pca_features[:, column2Y], c=data_labels.map(mydict))
    else:
        plt.scatter(data_features.iloc[:, column1X], data_features.iloc[:, column2Y], c=data_labels.map(mydict))
        plt.xlabel(data_features.columns[column1X])
        plt.ylabel(data_features.columns[column2Y])

    #legend1 = ax.legend(*scatter.legend_elements(),
    #                    loc="upper right", title="Classes")
    #ax.add_artist(legend1)

    fake_handles = [mpatches.Patch(color=item) for item in mydict.values()]
    label = mydict.keys()
    plt.legend(fake_handles, label, loc='upper right', prop={'size': 10})

    plt.show()
