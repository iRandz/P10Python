import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Prepare Data -------------------------------------------------------------

data = pd.read_csv("combined.csv", sep=';')

data_features = data.copy()
data_labels = data_features.pop('Type')

simplify = False
if simplify:
    for x in range(data_labels.size):
        if data_labels[x] == 'Assault' or data_labels[x] == 'Journey':
            data_labels[x] = 'Other'

# Force wrongly labelled data into other
for x in range(data_labels.size):
    y = data_labels[x]
    if y != 'Manage' and y != 'Journey' and y != 'Assault' and y != 'Other':
        data_labels[x] = 'Other'

data_features = np.array(data_features)

# Normalize to 0-1 range

min_max_scaler = preprocessing.MinMaxScaler()
data_features = min_max_scaler.fit_transform(data_features)

X_train, X_test, y_train, y_test = train_test_split(
        data_features, data_labels, test_size=0.2, random_state=42
    )

# Prepare and train model -------------------------------------------------------

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)

# Validate model -------------------------------------------------------------------

print(neigh.predict(X_test))

print(np.array(y_test))

print(neigh.score(X_test, y_test))
score = neigh.score(X_test, y_test)

# Display plots ---------------------------------------------------------------------------

show2D = 1
show3D = 0

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
    ax.scatter(data_features[:, column1X], data_features[:, column2Y], data_features[:, column3Z],
               c=data_labels.map(mydict), edgecolor="k")

    fake_handles = [mpatches.Patch(color=item) for item in mydict.values()]
    label = mydict.keys()
    ax.legend(fake_handles, label, loc='upper right', prop={'size': 10})

    ax.text(0, 0, 0, ("%.2f" % score).lstrip("0"))

    ax.set_xlabel('X Kills')
    ax.set_ylabel('Y Time')
    ax.set_zlabel('Z Shots')

    plt.show()

if show2D:
    column1X = 0
    column2Y = 2

    plt.scatter(data_features[:, column1X], data_features[:, column2Y], c=data_labels.map(mydict))
    #legend1 = ax.legend(*scatter.legend_elements(),
    #                    loc="upper right", title="Classes")
    #ax.add_artist(legend1)

    fake_handles = [mpatches.Patch(color=item) for item in mydict.values()]
    label = mydict.keys()
    plt.legend(fake_handles, label, loc='upper right', prop={'size': 10})

    plt.xlabel(data.columns[column1X + 1])
    plt.ylabel(data.columns[column2Y + 1])
    plt.show()
