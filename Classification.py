import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GroupShuffleSplit, StratifiedGroupKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

import Functions
import Settings

neighbors = 5  # KNN
nnHiddenLayers = (20, 10, 5)  # NN (10, 15, 6) (100, 5, 5): 70, (5, 5, 100): 66, (50, 5, 50): 66
nnSolver = 'lbfgs'  # NN (lbfgs, adam, sgd)
ovrEstimator = KNeighborsClassifier(n_neighbors=neighbors)  # OvR
svmKernel = 'rbf'  # SVM


def myFit(classifier, x_train, y_train, x_test, y_test, y_t, y_p):
	classifier.fit(x_train, y_train)
	score, unbalScore, y_true, y_predict = Functions.validate_classification_model(classifier, x_test, y_test, False)

	y_t = np.append(y_t, y_true)
	y_p = np.append(y_p, y_predict)

	return score, unbalScore, y_t, y_p


def classify(data_features, data_labels, print_stuff, settingsIn: Settings.Settings):
	# Create test/train split
	splits = 2
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
	unbalNeighList, unbalNeuralList, unbalOvrList, unbalSvmList = np.zeros(splits), np.zeros(splits), np.zeros(splits), np.zeros(splits)
	y_true_neigh, y_predict_neigh = np.empty(0), np.empty(0)
	y_true_neural, y_predict_neural = np.empty(0), np.empty(0)
	y_true_ovr, y_predict_ovr = np.empty(0), np.empty(0)
	y_true_svm, y_predict_svm = np.empty(0), np.empty(0)
	i = 0
	for train_idx, test_idx in sgkf.split(data_features, data_labels, settingsIn.groups):
		if print_stuff:
			print("")
			print("----------------------------")
			print("Train / Test Split: " + str(i))
			print("----------------------------")
		x_train, y_train = data_features.iloc[train_idx], data_labels.iloc[train_idx]
		x_test, y_test = data_features.iloc[test_idx], data_labels.iloc[test_idx]

		neighList[i], unbalNeighList[i], y_true_neigh, y_predict_neigh = myFit(neigh, x_train, y_train, x_test, y_test, y_true_neigh, y_predict_neigh)
		neuralList[i], unbalNeuralList[i], y_true_neural, y_predict_neural = myFit(neural, x_train, y_train, x_test, y_test, y_true_neural, y_predict_neural)
		ovrList[i], unbalOvrList[i], y_true_ovr, y_predict_ovr = myFit(ovr, x_train, y_train, x_test, y_test, y_true_ovr, y_predict_ovr)
		svmList[i], unbalSvmList[i], y_true_svm, y_predict_svm = myFit(svm_clf, x_train, y_train, x_test, y_test, y_true_svm, y_predict_svm)

		i += 1

	knn_score = np.mean(neighList)
	nn_score = np.mean(neuralList)
	ovr_score = np.mean(ovrList)
	svm_score = np.mean(svmList)

	knn_unbalScore = np.mean(unbalNeighList)
	nn_unbalScore = np.mean(unbalNeuralList)
	ovr_unbalScore = np.mean(unbalOvrList)
	svm_unbalScore = np.mean(unbalSvmList)

	# Validate model
	if print_stuff:
		print("\n -----------------------------")
		print("K nearest neighbor")
		print(knn_unbalScore)
		print(knn_score)
		print(confusion_matrix(y_true_neigh, y_predict_neigh))

	if print_stuff:
		print("\n -----------------------------")
		print("Multi layer perceptron")
		print("Iterations: " + str(neural.n_iter_))
		print("Layers: " + str(neural.hidden_layer_sizes))
		print(nn_unbalScore)
		print(nn_score)
		print(confusion_matrix(y_true_neural, y_predict_neural))

	if print_stuff:
		print("\n -----------------------------")
		print("One versus rest")
		print("Using: " + str(ovr.estimator))
		print(ovr_unbalScore)
		print(ovr_score)
		print(confusion_matrix(y_true_ovr, y_predict_ovr))

	if print_stuff:
		print("\n -----------------------------")
		print("SVM")
		print("Kernel: " + str(svm_clf.kernel))
		print(svm_unbalScore)
		print(svm_score)
		print(confusion_matrix(y_true_svm, y_predict_svm))

	return knn_score, nn_score, ovr_score, svm_score, knn_unbalScore, nn_unbalScore, ovr_unbalScore, svm_unbalScore
