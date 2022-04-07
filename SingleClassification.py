import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import Classification
import Functions


def SingleClassification(data_features, data_labels, settings):
	data_features = Functions.FeatureSelection(data_features, data_labels, settings.usePCA, settings.useFeatSel,
											   settings.dimensionalitySel, settings.dimensionalityPCA)
	knnScore, nnScore, ovrScore, svmScore = Classification.Classify(data_features, data_labels, True)

	# Display plots ---------------------------------------------------------------------------
	if settings.target == settings.ClassTarget.TYPE:
		mydict = {'Journey': 'green',
				  'Manage': 'blue',
				  'Assault': 'red',
				  'Other': 'black'}
	elif settings.target == settings.ClassTarget.GENDER:
		mydict = {'Male': 'blue',
				  'Female': 'red',
				  'Other': 'black'}
	else:
		sys.exit("Unknown target. Can't show plots." + settings.target.value)

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

		ax.text(0, 0, 0, ("%.2f" % nnScore).lstrip("0"))

		plt.show()

	if settings.show2D:

		plt.scatter(data_features.iloc[:, settings.column1X], data_features.iloc[:, settings.column2Y],
					c=data_labels.map(mydict))
		if not settings.usePCA:
			plt.xlabel(data_features.columns[settings.column1X])
			plt.ylabel(data_features.columns[settings.column2Y])

		fake_handles = [mpatches.Patch(color=item) for item in mydict.values()]
		label = mydict.keys()
		plt.legend(fake_handles, label, loc='upper right', prop={'size': 10})

		plt.show()
