import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

import Classification
import Functions


def dimensionality_check(data_features, data_labels):
    knn_scores = np.zeros(len(data_features.columns)-1)
    nn_scores = np.zeros(len(data_features.columns)-1)
    ovr_scores = np.zeros(len(data_features.columns)-1)
    svm_scores = np.zeros(len(data_features.columns)-1)

    for i in range(2, len(data_features.columns)-1):
        print(i)
        data_features_copy = pd.DataFrame(data_features.copy())
        working_features = Functions.feature_selection(data_features_copy, data_labels, False, True, i, 0)
        knn_scores[i], nn_scores[i], ovr_scores[i], svm_scores[i] = Classification.classify(working_features,
                                                                                            data_labels, False)

    def print_results(input_scores):
        # print(input_scores)
        print(input_scores.argmax())
        print(input_scores[input_scores.argmax()])

    print("Knn")
    print_results(knn_scores)
    print("---")
    print("NN")
    print_results(nn_scores)
    print("---")
    print("ovr")
    print_results(ovr_scores)
    print("---")
    print("SVM")
    print_results(svm_scores)

    t = np.arange(0., len(data_features.columns)-1, 1)

    fig, ax = plt.subplots()

    ax.plot(t, knn_scores, 'go', t, knn_scores, 'g')
    ax.plot(t, nn_scores, 'bo', t, nn_scores, 'b')
    ax.plot(t, ovr_scores, 'ro', t, ovr_scores, 'r')
    ax.plot(t, svm_scores, 'yo', t, svm_scores, 'y')

    knn = mpatches.Patch(color='green', label='knn')
    nn = mpatches.Patch(color='blue', label='nn')
    ovr = mpatches.Patch(color='red', label='ovr')
    svm = mpatches.Patch(color='yellow', label='svm')
    ax.legend(handles=[knn, nn, ovr, svm])
    plt.show()
