import numpy as np
import pandas as pd

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
        # print(inputScores)
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