from itertools import cycle

import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.metrics import auc, roc_curve, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize


def calcOvR(data_features, data_labels, settings):
	# ovrEstimator = KNeighborsClassifier(n_neighbors=5)  # OvR
	ovrEstimator = svm.SVC(kernel="linear", probability=True, random_state=0)
	ovr = OneVsRestClassifier(ovrEstimator)

	# y = label_binarize(data_labels, classes=['Objective', 'Exploration', ''])
	# y = label_binarize(data_labels, classes=
	# n_classes = 3

	X_train, X_test, y_train, y_test = train_test_split(data_features, data_labels, test_size=0.4, random_state=7)
	sgkf = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=1)
	for train_idx, test_idx in sgkf.split(data_features, data_labels, settings.groups):
		X_train, y_train = data_features.iloc[train_idx], data_labels.iloc[train_idx]
		X_test, y_test = data_features.iloc[test_idx], data_labels.iloc[test_idx]

	print("Classes in test set:")
	print(y_test.value_counts())
	print("---")

	# classMask = ['Objective', 'Exploration']
	classMask = ['Journey', 'Manage', 'Assault']
	y_test = label_binarize(y_test, classes=classMask)
	y_train = label_binarize(y_train, classes=classMask)

	n_classes = y_train.shape[1]

	ovr.fit(X_train, y_train)

	y_score = ovr.fit(X_train, y_train).predict_proba(X_test)

	prediction = ovr.predict(X_test)

	print(ovr)
	print(ovr.score(X_test, y_test))
	if prediction.ndim > 1:
		print(balanced_accuracy_score(y_test.argmax(axis=1), prediction.argmax(axis=1)))
		print(confusion_matrix(y_test.argmax(axis=1), prediction.argmax(axis=1)))
	else:
		print(balanced_accuracy_score(y_test, prediction))
		print(confusion_matrix(y_test, prediction))
		y_test = np.hstack((1 - y_test, y_test))
		n_classes = 2

	# Compute ROC curve and ROC area for each class
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	for i in range(n_classes):
		fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
		roc_auc[i] = auc(fpr[i], tpr[i])

	# Compute micro-average ROC curve and ROC area
	fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
	roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

	# Plot figure
	# First aggregate all false positive rates
	all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

	# Then interpolate all ROC curves at this points
	mean_tpr = np.zeros_like(all_fpr)
	for i in range(n_classes):
		mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

	# Finally average it and compute AUC
	mean_tpr /= n_classes

	fpr["macro"] = all_fpr
	tpr["macro"] = mean_tpr
	roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

	# Plot all ROC curves
	lw = 2
	plt.figure()
	plt.plot(
		fpr["micro"],
		tpr["micro"],
		label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
		color="deeppink",
		linestyle=":",
		linewidth=4,
	)

	plt.plot(
		fpr["macro"],
		tpr["macro"],
		label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
		color="navy",
		linestyle=":",
		linewidth=4,
	)

	colors = cycle(["aqua", "darkorange", "cornflowerblue"])
	for i, color in zip(range(n_classes), colors):
		plt.plot(
			fpr[i],
			tpr[i],
			color=color,
			lw=lw,
			label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
		)

	plt.plot([0, 1], [0, 1], "k--", lw=lw)
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel("False Positive Rate")
	plt.ylabel("True Positive Rate")
	plt.title(str(settings.classifier_target.value) + " : " + str(settings.dataFile))
	plt.legend(loc="lower right")
	plt.show()
