import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


def PlotAllFeatures(data_features, data_labels, settingsIn):
	# TODO Try the below link to implement better visualization of each feature
	# https://acaird.github.io/computer/python/datascience/2016/06/18/event-density
	for i in range(len(data_features.columns)):
		PlotSingleFeature(data_features.iloc[:, i], data_labels, settingsIn)


def PlotSingleFeature(data_feature: pd.DataFrame, data_labels, settingsIn):

	myDict = GetPlotcolorDict(settingsIn)
	feature = data_feature.name
	max_arg = data_feature.max()
	data_feature = np.split(data_feature, len(data_feature))

	plt.figure()
	plt.hlines(0, 0, max_arg)  # Draw a horizontal line
	plt.eventplot(data_feature, orientation='horizontal', colors=data_labels.map(myDict), lineoffsets=0)
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
