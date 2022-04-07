from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

import Functions

neighbors = 5  # KNN
nnHiddenLayers = (10, 15, 6)  # NN
ovrEstimator = KNeighborsClassifier(n_neighbors=neighbors)  # OvR
svmKernel = 'linear'  # SVM


def Classify(data_features, data_labels, printStuff):
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
	svmCLF.fit(X_train, y_train)

	# Validate model
	if printStuff:
		print("\n -----------------------------")
		print("K nearest neighbor")
	knnScore = Functions.ValidateModel(neigh, X_test, y_test, printStuff)

	if printStuff:
		print("\n -----------------------------")
		print("Multi layer perceptron")
		print("Iterations: " + str(neural.n_iter_))
		print("Layers: " + str(neural.hidden_layer_sizes))
	nnScore = Functions.ValidateModel(neural, X_test, y_test, printStuff)

	if printStuff:
		print("\n -----------------------------")
		print("One versus rest")
		print("Using: " + str(ovr.estimator))
	ovrScore = Functions.ValidateModel(ovr, X_test, y_test, printStuff)

	if printStuff:
		print("\n -----------------------------")
		print("SVM")
		print("Kernel: " + str(svmCLF.kernel))
	svmScore = Functions.ValidateModel(svmCLF, X_test, y_test, printStuff)

	return knnScore, nnScore, ovrScore, svmScore