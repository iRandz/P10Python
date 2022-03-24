import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

# Settings ----------------------------------------------------------------
show2D = 0
show3D = 1

usePCA = 1
dimensionalityPCA = 3
useFeatSel = 0
dimensionalitySel = 3

neighbors = 3  # KNN
nnHiddenLayers = (10, 15, 6)  # NN
ovrEstimator = KNeighborsClassifier(n_neighbors=neighbors)  # OvR
svmKernel = 'linear'  # SVM

# Prepare Data -------------------------------------------------------------
data = pd.read_csv("RealData.csv", sep=';')
#data = pd.read_csv("CombinedData.csv", sep=';')

#data.pop('Weekly playtime')
data.pop('Participant ID')
#data.pop('Age')
data.pop('Gender')
data.pop('Journey mean')
data.pop('Manage mean')
data.pop('Assault mean')
data.pop('Unique tiles')
#data.pop('Deaths')
#data.pop('MajorLore seen')
#data.pop('MajorLore reading time')
#data.pop('MajorLore close')
#data.pop('MajorLore interactions')

data_features = data.copy()
data_labels = data_features.pop('Type')

simplify = False
if simplify:
    for x in range(data_labels.size):
        if data_labels[x] == 'Assault' or data_labels[x] == 'Journey':
            data_labels[x] = 'Other'

# Preprocessing ----------------------------------------------------------
# Normalize to 0-1 range
min_max_scaler = preprocessing.MinMaxScaler()
data_features_norm = min_max_scaler.fit_transform(data_features)
for y in range(data_features.iloc[0, :].size):
    for x in range(data_features.iloc[:, 0].size):
        data_features.iloc[x, y] = data_features_norm[x, y]

# Feature extraction (PCA)
pca_features = PCA(n_components=dimensionalityPCA).fit_transform(data_features)
pca_Data = PCA(n_components=dimensionalityPCA)
pca_Data.fit(data_features)

#print("Components: " + str(pca_Data.components_))
print("Explained variance: " + str(pca_Data.explained_variance_))
print("Explained variance ratio: " + str(pca_Data.explained_variance_ratio_))

# Feature selection
selector = SelectKBest(chi2, k=dimensionalitySel) #.fit_transform(data_features, data_labels)
selector.fit(data_features, data_labels)
cols = selector.get_support(indices=True)
data_features = data_features.iloc[:, cols]

if usePCA:
    data_features = pca_features
    if useFeatSel:
        selector = SelectKBest(k=dimensionalitySel)
        selector.fit(pca_features, data_labels)
        cols = selector.get_support(indices=True)
        data_features = pca_features[:, cols]
else:
    print(data_features.columns)

# Force wrongly labelled data into 'Other'
for x in range(data_labels.size):
    y = data_labels[x]
    if y != 'Manage' and y != 'Journey' and y != 'Assault' and y != 'Other':
        data_labels[x] = 'Other'

# Create test/train split
X_train, X_test, y_train, y_test = train_test_split(
        data_features, data_labels, test_size=0.4, random_state=42
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
svmCLF.fit(X_train, y_train)

# Validate model
print("K nearest neighbor")
print(neigh.predict(X_test))
print(np.array(y_test))
print(neigh.score(X_test, y_test))

print("\n -----------------------------")
print("Multi layer perceptron")
print("Iterations: " + str(neural.n_iter_))
print("Layers: " + str(neural.hidden_layer_sizes))
print(neural.score(X_test, y_test))

print("\n -----------------------------")
print("One versus rest")
print("Using: " + str(ovr.estimator))
print(ovr.score(X_test, y_test))

print("\n -----------------------------")
print("SVM")
print("Kernel: " + str(svmCLF.kernel))
print(svmCLF.score(X_test, y_test))

score = neigh.score(X_test, y_test)

# Display plots ---------------------------------------------------------------------------
mydict = {'Journey': 'green',
          'Manage': 'blue',
          'Assault': 'red',
          'Other': 'black'}

if show3D:

    column1X = 0
    column2Y = 1
    column3Z = 2

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
    column1X = 0
    column2Y = 1

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
