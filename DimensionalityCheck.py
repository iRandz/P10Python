import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

import Classification
import Functions
import Settings


def dimensionality_check(data_features, data_labels, settingsIn: Settings.Settings):
    knn_scores = np.zeros(len(data_features.columns)-1)
    nn_scores = np.zeros(len(data_features.columns)-1)
    ovr_scores = np.zeros(len(data_features.columns)-1)
    svm_scores = np.zeros(len(data_features.columns)-1)

    knn_unbalScores = np.zeros(len(data_features.columns) - 1)
    nn_unbalScores = np.zeros(len(data_features.columns) - 1)
    ovr_unbalScores = np.zeros(len(data_features.columns) - 1)
    svm_unbalScores = np.zeros(len(data_features.columns) - 1)

    featureMemory = None
    for i in range(1, len(data_features.columns)-1):
        print(i)
        data_features_copy = pd.DataFrame(data_features.copy())
        FeatSelo = True
        working_features = Functions.feature_selection(data_features_copy, data_labels, not FeatSelo, FeatSelo, i, i)
        if i is 1:
            featureMemory = working_features.columns
            print(featureMemory)
            continue
        print(working_features.columns.copy().drop(featureMemory).values)
        featureMemory = working_features.columns
        knn_scores[i], nn_scores[i], ovr_scores[i], svm_scores[i], \
            knn_unbalScores[i], nn_unbalScores[i], ovr_unbalScores[i], svm_unbalScores[i] = Classification.classify(working_features,
                                                                                            data_labels, False, settingsIn)

    def print_results(input_scores, unbalInput_scores):
        # print(input_scores)
        dimensionality = unbalInput_scores.argmax()
        print(dimensionality)
        print("Balanced Accuracy: " + str(input_scores[dimensionality]) + ", Accuracy: " + str(unbalInput_scores[dimensionality]))

        dimensionality = input_scores.argmax()
        print(dimensionality)
        print("Balanced Accuracy: " + str(input_scores[dimensionality]) + ", Accuracy: " + str(unbalInput_scores[dimensionality]))

    print("Knn")
    print_results(knn_scores, knn_unbalScores)
    print("---")
    print("NN")
    print_results(nn_scores, nn_unbalScores)
    print("---")
    print("ovr")
    print_results(ovr_scores, ovr_unbalScores)
    print("---")
    print("SVM")
    print_results(svm_scores, svm_unbalScores)

    t = np.arange(0., len(data_features.columns)-1, 1)

    fig, ax = plt.subplots()

    ax.plot(t, knn_unbalScores, 'g--^', alpha=0.5)
    ax.plot(t, nn_unbalScores, 'b--^', alpha=0.5)
    ax.plot(t, ovr_unbalScores, 'r--^', alpha=0.5)
    ax.plot(t, svm_unbalScores, 'y--^', alpha=0.5)

    ax.plot(t, knn_scores, 'go', t, knn_scores, 'g')
    ax.plot(t, nn_scores, 'bo', t, nn_scores, 'b')
    ax.plot(t, ovr_scores, 'ro', t, ovr_scores, 'r')
    ax.plot(t, svm_scores, 'yo', t, svm_scores, 'y')

    knn = mpatches.Patch(color='green', label='knn')
    nn = mpatches.Patch(color='blue', label='nn')
    ovr = mpatches.Patch(color='red', label='ovr')
    svm = mpatches.Patch(color='yellow', label='svm')
    ax.legend(handles=[knn, nn, ovr, svm])
    plt.title(str(settingsIn.classifier_target.value) + " : " + str(settingsIn.dataFile))
    plt.grid(True, 'major', 'y', linewidth=2)
    plt.ylim([0.3, 1])
    plt.show()
