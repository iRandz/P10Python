import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc


def PlotAllFeatures(data_features, data_labels, settingsIn):
	for i in range(len(data_features.columns)):
		PlotSingleFeatureHist(data_features.iloc[:, i], data_labels, settingsIn)


def PlotSingleFeatureHist(data_feature, data_labels, settingsIn):

	labels = data_labels.unique()
	df = np.empty(3, dtype=object)
	counts = np.empty(3, dtype=object)
	bins = np.empty(3, dtype=object)

	fig, ax = plt.subplots()
	valMin = np.min(data_feature)
	valMax = np.max(data_feature)
	for i in range(len(labels)):
		mask = data_labels == labels[i]
		df[i] = data_feature[mask]
		counts[i], bins[i] = np.histogram(df[i], 8, (valMin, valMax))
		currentBin = bins[i]
		counts[i] = counts[i] / len(df[i])
		ax.plot(currentBin[:-1], counts[i], label=labels[i])

	handles, labels = ax.get_legend_handles_labels()
	ax.legend(handles, labels)
	plt.xlabel(data_feature.name)
	plt.show()


def EventPlotSingleFeature(data_feature, data_labels, settingsIn):

	myDict = GetPlotcolorDict(settingsIn)

	feature = data_feature.name
	max_arg = data_feature.max()
	data_feature = np.split(data_feature, len(data_feature))
	offsetDict = {'Journey': '0',
			  'Manage': '1',
			  'Assault': '2',
			  'Other': '3'}

	plt.figure()
	plt.hlines(0, 0, max_arg)  # Draw a horizontal line
	plt.eventplot(data_feature, orientation='horizontal', colors=data_labels.map(myDict), lineoffsets=data_labels.map(offsetDict))
	plt.axis("on")
	plt.xlabel(feature)
	plt.show()


def GetPlotcolorDict(settings):
	if settings.classifier_target == settings.ClassTarget.TYPE:
		mydict = {'Journey': 'g',
				  'Manage': 'b',
				  'Assault': 'r',
				  'Other': 'y'}
	elif settings.classifier_target == settings.ClassTarget.GENDER:
		mydict = {'Male': 'blue',
				  'Female': 'red',
				  'Other': 'black'}
	elif settings.classifier_target == settings.ClassTarget.OBJOREXP:
		mydict = {'Objective': 'red',
				  'Exploration': 'green'}
	else:
		sys.exit("Unknown target. Can't show plots." + settings.classifier_target.value)

	return mydict
