import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import Classification
import Functions


def SingelClassification(data_features, data_labels, mainML):

	data_features = Functions.FeatureSelection(data_features, data_labels, mainML.usePCA, mainML.useFeatSel, mainML.dimensionalitySel, mainML.dimensionalityPCA)
	knnScore, nnScore, ovrScore, svmScore = Classification.Classify(data_features, data_labels, True)

	# Display plots ---------------------------------------------------------------------------
	if mainML.target == mainML.ClassTarget.TYPE:
		mydict = {'Journey': 'green',
				  'Manage': 'blue',
				  'Assault': 'red',
				  'Other': 'black'}
	elif mainML.target == mainML.ClassTarget.GENDER:
		mydict = {'Male': 'blue',
				  'Female': 'red',
				  'Other': 'black'}
	else:
		sys.exit("Unknown target. Can't show plots." + mainML.target.value)

	if mainML.show3D:
		fig = plt.figure(1, figsize=(8, 6))
		ax = fig.add_subplot(projection='3d')

		ax.scatter(data_features.iloc[:, mainML.column1X], data_features.iloc[:, mainML.column2Y], data_features.iloc[:, mainML.column3Z],
				   c=data_labels.map(mydict), edgecolor="k")
		if not mainML.usePCA:
			ax.set_xlabel(data_features.columns[mainML.column1X])
			ax.set_ylabel(data_features.columns[mainML.column2Y])
			ax.set_zlabel(data_features.columns[mainML.column3Z])

		fake_handles = [mpatches.Patch(color=item) for item in mydict.values()]
		label = mydict.keys()
		ax.legend(fake_handles, label, loc='upper right', prop={'size': 10})

		ax.text(0, 0, 0, ("%.2f" % nnScore).lstrip("0"))

		plt.show()

	if mainML.show2D:

		plt.scatter(data_features.iloc[:, mainML.column1X], data_features.iloc[:, mainML.column2Y], c=data_labels.map(mydict))
		if not mainML.usePCA:
			plt.xlabel(data_features.columns[mainML.column1X])
			plt.ylabel(data_features.columns[mainML.column2Y])

		fake_handles = [mpatches.Patch(color=item) for item in mydict.values()]
		label = mydict.keys()
		plt.legend(fake_handles, label, loc='upper right', prop={'size': 10})

		plt.show()
