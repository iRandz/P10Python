import numpy as np
import pandas as pd

import Functions
import Settings
import SingleClassification
import FeatureDict as fD


if False:
    settings = Settings.Settings()
    # Prepare Data -------------------------------------------------------------
    data = pd.read_csv("Data/CombinedSecondsLog.csv", sep=';')

    print("Check for null:")
    print(np.any(pd.isnull(data)))
    print("---")

    # Calculate 2nd order features
    data_features: pd.DataFrame = Functions.calc_derived_features(data)

    # Process data
    data_features, data_labels = Functions.process_data(data_features, settings)
    data_features_day = pd.DataFrame(data_features.pop(fD.DAY))
    data_features_Difference: pd.DataFrame = data_features.copy(True)
    for i in range(data_features.iloc[:, 0].size):
        if i == 1:
            continue
        currentRow = data_features.iloc[i]
        previousRow = data_features.iloc[i-1]
        data_features_Difference.iloc[i] = currentRow-previousRow

    data_features_Difference.to_csv('Differentieret_CombinedSecondsLog.csv', sep=';', index=False)

    # Running average
    runningRange = 10

    data_features_running: pd.DataFrame = data_features_Difference.rolling(runningRange).sum()
    prevDayString = "4"
    for i in range(data_features_running.iloc[:, 0].size, 0, -1):
        rowInfo = data_features_running.iloc[i-1]
        dayRow = data_features_day.iloc[i-1]
        currentDayString = str(dayRow[fD.DAY])
        if currentDayString == prevDayString:
            continue
        elif currentDayString != prevDayString:
            dropList = range(i, i + runningRange - 1)
            data_features_running = data_features_running.drop(dropList)
            data_labels = data_labels.drop(dropList)
            prevDayString = currentDayString

    dropList = range(0, runningRange - 1)
    data_features_running = data_features_running.drop(dropList)
    data_labels = data_labels.drop(dropList)

    data_features_running.to_csv('Running_Differentieret_CombinedSecondsLog.csv', sep=';', index=False)
    data_labels.to_csv('Running_Lables_CombinedSecondsLog.csv', sep=';', index=False)

    print("---/n Data processing done")

if __name__ == '__main__':
    print("Only Main")
    settings = Settings.Settings()

    data_features_running = pd.read_csv("Running_Differentieret_CombinedSecondsLog.csv", sep=';')
    data_labels = pd.read_csv("Running_Lables_CombinedSecondsLog.csv", sep=';')

    print("Check for null:")
    print(np.any(pd.isnull(data_features_running)))
    print("---")

    data_features_running = Functions.normalize(data_features_running)
    print("Running classification")
    SingleClassification.single_classification(data_features_running, data_labels, settings)
