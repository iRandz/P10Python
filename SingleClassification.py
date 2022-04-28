import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import Classification
import FeatureDict
import Functions
import Settings


def single_classification(data_features, data_labels, settings: Settings.Settings):
	data_features = Functions.feature_selection(data_features, data_labels, settings.usePCA, settings.useFeatSel,
												settings.dimensionalitySel, settings.dimensionalityPCA)

	print(data_features.columns)

	# data_features = data_features[[FeatureDict.AGE, FeatureDict.MLT_PER, FeatureDict.TIMESS, FeatureDict.ML_READINGTIME]]

	knn_score, nn_score, ovr_score, svm_score = Classification.classify(data_features, data_labels, True, settings)

	print("---")
	print("Plotting:")
	# Display plots ---------------------------------------------------------------------------
	if settings.classifier_target == settings.ClassTarget.TYPE:
		mydict = {'Journey': 'green',
				  'Manage': 'blue',
				  'Assault': 'red',
				  'Other': 'black'}
	elif settings.classifier_target == settings.ClassTarget.GENDER:
		mydict = {'Male': 'blue',
				  'Female': 'red',
				  'Other': 'black'}
	elif settings.classifier_target == settings.ClassTarget.OBJOREXP:
		mydict = {'Objective': 'red',
				  'Exploration': 'green'}
	else:
		sys.exit("Unknown target. Can't show plots." + settings.classifier_target.value)

	if settings.show3D:
		fig = plt.figure(1, figsize=(8, 6))
		ax = fig.add_subplot(projection='3d')

		ax.scatter(data_features.iloc[:, settings.column1X], data_features.iloc[:, settings.column2Y],
				   data_features.iloc[:, settings.column3Z],
				   c=data_labels.map(mydict), edgecolor="k")
		if not settings.usePCA:
			ax.set_xlabel(data_features.columns[settings.column1X])
			ax.set_ylabel(data_features.columns[settings.column2Y])
			ax.set_zlabel(data_features.columns[settings.column3Z])

		fake_handles = [mpatches.Patch(color=item) for item in mydict.values()]
		label = mydict.keys()
		ax.legend(fake_handles, label, loc='upper right', prop={'size': 10})

		ax.text(0, 0, 0, ("%.2f" % nn_score).lstrip("0"))

		plt.title(str(settings.classifier_target.value) + " : " + str(settings.dataFile))

		plt.show()

	if settings.show2D:

		plt.scatter(data_features.iloc[:, settings.column1X], data_features.iloc[:, settings.column2Y],
					c=data_labels.map(mydict), alpha=0.3)
		if not settings.usePCA:
			plt.xlabel(data_features.columns[settings.column1X])
			plt.ylabel(data_features.columns[settings.column2Y])

		fake_handles = [mpatches.Patch(color=item) for item in mydict.values()]
		label = mydict.keys()
		plt.legend(fake_handles, label, loc='upper right', prop={'size': 10})

		plt.title(str(settings.classifier_target.value) + " : " + str(settings.dataFile))

		plt.show()
