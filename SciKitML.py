import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA

# Settings ----------------------------------------------------------------
usePCA = 1
show2D = 0
show3D = 1
dimensionality = 3
neighbors = 3

# Prepare Data -------------------------------------------------------------
#data = pd.read_csv("CombinedData.csv", sep=';')
data = pd.read_csv("RealData.csv", sep=';')

#data.pop('Weekly playtime')
data.pop('Participant ID')
#data.pop('Age')
data.pop('Gender')
data.pop('Journey mean')
data.pop('Manage mean')
data.pop('Assault mean')
data.pop('Unique tiles')
#data.pop('Deaths')
data.pop('MajorLore seen')
data.pop('MajorLore reading time')
data.pop('MajorLore close')
data.pop('MajorLore interactions')

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
    for x in range(data_features.iloc[:,0].size):
        data_features.iloc[x, y] = data_features_norm[x, y]

# Feature extraction (PCA)
pca_features = PCA(n_components=dimensionality).fit_transform(data_features)

# Feature selection
selector = SelectKBest(chi2, k=dimensionality) #.fit_transform(data_features, data_labels)
selector.fit(data_features, data_labels)
cols = selector.get_support(indices=True)
data_features = data_features.iloc[:, cols]

if usePCA:
    data_features = pca_features
else:
    print(data_features.columns)

# Force wrongly labelled data into 'Other'
for x in range(data_labels.size):
    y = data_labels[x]
    if y != 'Manage' and y != 'Journey' and y != 'Assault' and y != 'Other':
        data_labels[x] = 'Other'

# Create test/train split
X_train, X_test, y_train, y_test = train_test_split(
        data_features, data_labels, test_size=0.2, random_state=42
    )

# Prepare and train model -------------------------------------------------------
neigh = KNeighborsClassifier(n_neighbors=neighbors)
neigh.fit(X_train, y_train)

# Validate model -------------------------------------------------------------------
print(neigh.predict(X_test))
print(np.array(y_test))
print(neigh.score(X_test, y_test))

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
