import sys
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import Functions
import Regression


def single_classification(data_features, data_labels, settings):
	data_features = Functions.feature_selection(data_features, data_labels, settings.usePCA, settings.useFeatSel, 1, 1)

	# data_features = pd.DataFrame(data_features['Shots fired'])

	# Create test/train split
	x_train, x_test, y_train, y_test = train_test_split(
		data_features, data_labels, test_size=0.2, random_state=42
	)

	nn_predictions = Regression.regress(x_train, y_train, x_test, y_test, True)

	# Display plots ---------------------------------------------------------------------------
	if settings.show3D:
		print("Sad story bro")

	if settings.show2D:
		order = np.argsort(x_test.iloc[:, 0])
		xs = np.array(nn_predictions)[order]
		ys = np.array(x_test)[order]

		plt.scatter(x_train, y_train, color = "green", alpha=0.2, label = "Training set")
		plt.scatter(x_test, y_test, color="red", alpha=0.5, label="Test set, ground truth")
		plt.scatter(x_test, nn_predictions, alpha=0.3, color="blue", label="Test set, predictions")
		plt.plot(ys[:, 0], xs, color="blue", linewidth=1)

		plt.xlabel(data_features.iloc[:, 0].name)
		plt.ylabel(data_labels.name)

		plt.grid(b=True, color="grey", alpha=0.2)

		plt.legend()
		plt.show()