import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

import Classification
import Functions


def DimensionalityCheck(data_features, data_labels):
    knnScores = np.zeros(len(data_features.columns)-1)
    nnScores = np.zeros(len(data_features.columns)-1)
    ovrScores = np.zeros(len(data_features.columns)-1)
    svmScores = np.zeros(len(data_features.columns)-1)

    for i in range(2, len(data_features.columns)-1):
        data_features_copy = pd.DataFrame(data_features.copy())
        working_features = Functions.FeatureSelection(data_features_copy, data_labels, False, True, i, 0)
        knnScores[i], nnScores[i], ovrScores[i], svmScores[i] = Classification.Classify(working_features, data_labels, False)


    def PrintResults(inputScores):
        print(inputScores)
        print(inputScores.argmax())
        print(inputScores[inputScores.argmax()])


    print("Knn")
    PrintResults(knnScores)
    print("NN")
    PrintResults(nnScores)
    print("ovr")
    PrintResults(ovrScores)
    print("SVM")
    PrintResults(svmScores)

    t = np.arange(0., len(data_features.columns)-1, 1)

    fig, ax = plt.subplots()

    ax.plot(t, knnScores, 'go', t, knnScores, 'g')
    ax.plot(t, nnScores, 'bo', t, nnScores, 'b')
    ax.plot(t, ovrScores, 'ro', t, ovrScores, 'r')
    ax.plot(t, svmScores, 'yo', t, svmScores, 'y')

    knn = mpatches.Patch(color='green', label='knn')
    nn = mpatches.Patch(color='blue', label='nn')
    ovr = mpatches.Patch(color='red', label='ovr')
    svm = mpatches.Patch(color='yellow', label='svm')
    ax.legend(handles=[knn, nn, ovr, svm])
    plt.show()
