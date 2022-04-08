from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

import Functions

neighbors = 5  # KNN
nnHiddenLayers = (10, 15, 6)  # NN
nnSolver = 'lbfgs'  # NN (lbfgs, adam, sgd)
ovrEstimator = KNeighborsClassifier(n_neighbors=neighbors)  # OvR
svmKernel = 'linear'  # SVM


def classify(data_features, data_labels, print_stuff):
	# Create test/train split
	x_train, x_test, y_train, y_test = train_test_split(
			data_features, data_labels, test_size=0.2, random_state=42
		)

	# ML -----------------------------------------------------------------------------------------
	# Prepare and train model
	neigh = KNeighborsClassifier(n_neighbors=neighbors)
	neigh.fit(x_train, y_train)

	neural = MLPClassifier(solver=nnSolver, alpha=1e-5, hidden_layer_sizes=nnHiddenLayers, random_state=1, max_iter=15000)
	neural.fit(x_train, y_train)

	ovr = OneVsRestClassifier(ovrEstimator)
	ovr.fit(x_train, y_train)

	svm_clf = svm.SVC(kernel=svmKernel)
	svm_clf.fit(x_train, y_train)

	# Validate model
	if print_stuff:
		print("\n -----------------------------")
		print("K nearest neighbor")
	knn_score = Functions.validate_classification_model(neigh, x_test, y_test, print_stuff)

	if print_stuff:
		print("\n -----------------------------")
		print("Multi layer perceptron")
		print("Iterations: " + str(neural.n_iter_))
		print("Layers: " + str(neural.hidden_layer_sizes))
	nn_score = Functions.validate_classification_model(neural, x_test, y_test, print_stuff)

	if print_stuff:
		print("\n -----------------------------")
		print("One versus rest")
		print("Using: " + str(ovr.estimator))
	ovr_score = Functions.validate_classification_model(ovr, x_test, y_test, print_stuff)

	if print_stuff:
		print("\n -----------------------------")
		print("SVM")
		print("Kernel: " + str(svm_clf.kernel))
	svm_score = Functions.validate_classification_model(svm_clf, x_test, y_test, print_stuff)

	return knn_score, nn_score, ovr_score, svm_score
