import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split, GroupShuffleSplit, StratifiedGroupKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

import Functions
import Settings

neighbors = 5  # KNN
nnHiddenLayers = (50, 5, 5)  # NN (10, 15, 6) (100, 5, 5): 70, (5, 5, 100): 66, (50, 5, 50): 66
nnSolver = 'lbfgs'  # NN (lbfgs, adam, sgd)
ovrEstimator = KNeighborsClassifier(n_neighbors=neighbors)  # OvR
svmKernel = 'linear'  # SVM


def myFit(classifier, x_train, y_train, x_test, y_test, print):
	classifier.fit(x_train, y_train)
	score = Functions.validate_classification_model(classifier, x_test, y_test, print)
	return score


def classify(data_features, data_labels, print_stuff, settingsIn: Settings.Settings):
	# Create test/train split
	splits = 5
	sgkf = StratifiedGroupKFold(n_splits=splits, shuffle=True, random_state=42)
	# gss = GroupShuffleSplit(n_splits=5, train_size=.6, random_state=42)
	# train_idx, test_idx = None, None

	# Prepare models
	neigh = KNeighborsClassifier(n_neighbors=neighbors)
	neural = MLPClassifier(solver=nnSolver, alpha=1e-5, hidden_layer_sizes=nnHiddenLayers, random_state=1,
						   max_iter=15000)
	ovr = OneVsRestClassifier(ovrEstimator)
	svm_clf = svm.SVC(kernel=svmKernel)

	# Cross validation
	neighList, neuralList, ovrList, svmList = np.zeros(splits), np.zeros(splits), np.zeros(splits), np.zeros(splits)
	i = 0
	for train_idx, test_idx in sgkf.split(data_features, data_labels, settingsIn.groups):
		if print_stuff:
			print("")
			print("----------------------------")
			print("Train / Test Split: " + str(i))
			print("----------------------------")
		x_train, y_train = data_features.iloc[train_idx], data_labels.iloc[train_idx]
		x_test, y_test = data_features.iloc[test_idx], data_labels.iloc[test_idx]

		neighList[i] = myFit(neigh, x_train, y_train, x_test, y_test, print_stuff)
		neuralList[i] = myFit(neural, x_train, y_train, x_test, y_test, print_stuff)
		ovrList[i] = myFit(ovr, x_train, y_train, x_test, y_test, print_stuff)
		svmList[i] = myFit(svm_clf, x_train, y_train, x_test, y_test, print_stuff)

		i += 1

	knn_score = np.mean(neighList)
	nn_score = np.mean(neuralList)
	ovr_score = np.mean(ovrList)
	svm_score = np.mean(svmList)

	# Validate model
	if print_stuff:
		print("\n -----------------------------")
		print("K nearest neighbor")
		print(knn_score)

	if print_stuff:
		print("\n -----------------------------")
		print("Multi layer perceptron")
		print("Iterations: " + str(neural.n_iter_))
		print("Layers: " + str(neural.hidden_layer_sizes))
		print(nn_score)

	if print_stuff:
		print("\n -----------------------------")
		print("One versus rest")
		print("Using: " + str(ovr.estimator))
		print(ovr_score)

	if print_stuff:
		print("\n -----------------------------")
		print("SVM")
		print("Kernel: " + str(svm_clf.kernel))
		print(svm_score)

	return knn_score, nn_score, ovr_score, svm_score
