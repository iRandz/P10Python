import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import FeatureDict
import Functions
import Settings

settings = Settings.Settings()
# Prepare Data -------------------------------------------------------------
file = "Data/CombinedDayLog.CSV"
data = pd.read_csv(file, sep=';')

settings.dataFile = file

print("Check for null:")
print(np.any(pd.isnull(data)))
print("---")

# Calculate 2nd order features
data_features = Functions.calc_derived_features(data)

# Process data
# data_features, data_labels = Functions.process_data(data_features, settings)

# Select features
featureList = [FeatureDict.DAY]
data_features = pd.DataFrame([data_features.pop(FeatureDict.L_READINGTIME), data_features.pop(FeatureDict.LORE), data_features.pop(FeatureDict.LT_PER)])
data_features = data_features.swapaxes(0, 1)
FeatSelo = True
#data_features = Functions.feature_selection(data_features, data_labels, not FeatSelo, FeatSelo, 3, 3)
print(data_features.columns.values)
data_feature_labels = data_features.copy()

# Standardize
if settings.normalize:
    data_features = Functions.normalize(data_features)

data_features = np.array(data_features)

# #############################################################################
# Compute DBSCAN
db = DBSCAN(eps=0.1, min_samples=10).fit(data_features)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)
#print("Homogeneity: %0.3f" % metrics.homogeneity_score(data_labels, labels))
#print("Completeness: %0.3f" % metrics.completeness_score(data_labels, labels))
#print("V-measure: %0.3f" % metrics.v_measure_score(data_labels, labels))
#print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(data_labels, labels))
#print(
#    "Adjusted Mutual Information: %0.3f"
#    % metrics.adjusted_mutual_info_score(data_labels, labels)
#)
#print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(data_features, labels))

# #############################################################################
# Plot result
# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
fig = plt.figure(1, figsize=(8, 6))
ax = fig.add_subplot(projection='3d')
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = labels == k

    xy = data_features[class_member_mask & core_samples_mask]
    ax.plot(
        xy[:, 0],
        xy[:, 1],
        xy[:, 2],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=14,
    )

    xy = data_features[class_member_mask & ~core_samples_mask]
    ax.plot(
        xy[:, 0],
        xy[:, 1],
        xy[:, 2],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=6,
    )

ax.set_xlabel(data_feature_labels.columns[settings.column1X])
ax.set_ylabel(data_feature_labels.columns[settings.column2Y])
ax.set_zlabel(data_feature_labels.columns[settings.column3Z])

plt.title("Estimated number of clusters: %d" % n_clusters_)
plt.show()
