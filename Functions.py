import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

import FeatureDict as fD
import Plotting
import Settings


def recalc_manage(row):
    if row['Type'] == 'Manage':
        if row['Assault mean'] > row['Journey mean']:
            return 'Assault'
        return 'Journey'
    return row['Type']


def recalc_journey(row):
    if row['Type'] == 'Journey':
        if row['Assault mean'] > row['Manage mean']:
            return 'Assault'
        return 'Manage'
    return row['Type']


def calc_ratio(row, timeName, interactionName):
    if row[interactionName] == 0:
        return 0
    return row[timeName] / row[interactionName]


def sumTotal(row, normal, major):
    return row[normal] + row[major]


def calc_derived_features(dataframe):
    dataframe[fD.MLT_PER] = dataframe.apply(
        lambda row: calc_ratio(row, 'MajorLore reading time', 'MajorLore interactions'), axis=1)
    dataframe[fD.LT_PER] = dataframe.apply(
        lambda row: calc_ratio(row, 'Lore reading time', 'Lore interactions'), axis=1)
    # dataframe['MajorLoreRatioClose'] = dataframe.apply (lambda row: CalcRatio(row, 'MajorLore interactions',
    # 'MajorLore seen'), axis=1)
    dataframe[fD.LR_SEEN] = dataframe.apply(
        lambda row: calc_ratio(row, 'Lore interactions', 'Lore seen'), axis=1)
    dataframe[fD.LR_CLOSE] = dataframe.apply(
        lambda row: calc_ratio(row, 'Lore interactions', 'Lore close'), axis=1)
    dataframe[fD.RR_SEEN] = dataframe.apply(
        lambda row: calc_ratio(row, 'Resources', 'Resources seen '), axis=1)
    dataframe[fD.RR_CLOSE] = dataframe.apply(
        lambda row: calc_ratio(row, 'Resources', 'Resources close'), axis=1)
    dataframe[fD.ER_SEEN] = dataframe.apply(lambda row: calc_ratio(row, 'Kills', 'Enemies seen'), axis=1)
    dataframe[fD.ER_CLOSE] = dataframe.apply(
        lambda row: calc_ratio(row, 'Kills', 'Enemies close'), axis=1)
    dataframe[fD.MT_PER] = dataframe.apply(lambda row: calc_ratio(row, 'MapTime', 'Opened map'), axis=1)

    dataframe[fD.TOTAL_KILLS] = dataframe.apply(lambda row: sumTotal(row, fD.KILLS, fD.M_KILLS), axis=1)
    dataframe[fD.TOTAL_L_INTERACTIONS] = dataframe.apply(lambda row: sumTotal(row, fD.LORE, fD.ML_INTERACTIONS), axis=1)
    dataframe[fD.TOTAL_L_SEEN] = dataframe.apply(lambda row: sumTotal(row, fD.L_SEEN, fD.ML_SEEN), axis=1)
    dataframe[fD.TOTAL_L_CLOSE] = dataframe.apply(lambda row: sumTotal(row, fD.L_CLOSE, fD.ML_CLOSE), axis=1)
    dataframe[fD.TOTAL_L_READING] = dataframe.apply(lambda row: sumTotal(row, fD.L_READINGTIME, fD.ML_READINGTIME), axis=1)
    dataframe[fD.TOTAL_E_SEEN] = dataframe.apply(lambda row: sumTotal(row, fD.E_SEEN, fD.ME_SEEN), axis=1)
    dataframe[fD.TOTAL_E_CLOSE] = dataframe.apply(lambda row: sumTotal(row, fD.E_CLOSE, fD.ME_CLOSE), axis=1)
    dataframe[fD.TOTAL_RESOURCES] = dataframe.apply(lambda row: sumTotal(row, fD.RESOURCES, fD.M_RESOURCES), axis=1)
    dataframe[fD.TOTAL_R_SEEN] = dataframe.apply(lambda row: sumTotal(row, fD.R_SEEN, fD.MR_SEEN), axis=1)
    dataframe[fD.TOTAL_R_CLOSE] = dataframe.apply(lambda row: sumTotal(row, fD.R_CLOSE, fD.MR_CLOSE), axis=1)

    return dataframe


def process_data(data, settingsIn: Settings.Settings):
    recalcManage = settingsIn.recalcManage
    removeManage = settingsIn.removeManage
    if settingsIn.test == settingsIn.CurrentTest.DIMENSIONALITY or settingsIn.test == settingsIn.CurrentTest.CLASSIFICATION:
        target = settingsIn.classifier_target
    elif settingsIn.test == settingsIn.CurrentTest.REGRESSION:
        target = settingsIn.regressor_target
    elif settingsIn.test == settingsIn.CurrentTest.ROC:
        target = settingsIn.classifier_target
    else:
        # Do nothing
        print("What u doing?")
        return

    def safe_pop(data_to_pop_from, popValue):
        if popValue in data_to_pop_from.columns:
            data_to_pop_from.pop(popValue)

    if recalcManage:
        data['Type'] = data.apply(lambda row: recalc_manage(row), axis=1)

    if settingsIn.recalcJourney:
        data['Type'] = data.apply(lambda row: recalc_journey(row), axis=1)

    if removeManage:
        data = data[data['Type'] != 'Manage']

    data = handle_invalid_data(settingsIn, data)

    printSumVariables = False
    if printSumVariables:
        CheckVariable: pd.DataFrame = data[fD.AGE]
        print(CheckVariable.max())
        print(CheckVariable.min())
        print(CheckVariable.std())
        print(CheckVariable.mean())

    data_labels = data.pop(target.value)

    safe_pop(data, 'Type')
    safe_pop(data, 'Gender')

    # data.pop('Weekly playtime')
    settingsIn.groups = data.pop('Participant ID')
    safe_pop(data, fD.AGE)
    data.pop('Journey mean')
    data.pop('Manage mean')
    data.pop('Assault mean')
    safe_pop(data, 'Obj or exp')
    safe_pop(data, 'Previous participant')
    safe_pop(data, fD.PLAYTIME)
    # data.pop('Major enemies close')
    # data.pop('Major kills')
    # data.pop(FeatureDict.E_SEEN)
    # data.pop(FeatureDict.ME_SEEN)
    # data.pop('Opened stats')
    # data.pop('Unique tiles')
    # data.pop('Deaths')
    # data.pop('MajorLore seen')
    # data.pop('MajorLore reading time')
    # data.pop('MajorLore close')
    # data.pop('MajorLore interactions')
    # data.pop('Major resources close')
    # data.pop('Major resources seen ')
    # data.pop('Lore interactions')
    # data.pop(FeatureDict.LT_PER)
    # data.pop(FeatureDict.L_READINGTIME)
    # data.pop(FeatureDict.MOUSE_MOVE)
    # data.pop(FeatureDict.FAR)

    print("Constant columns:")
    print(data.loc[:, (data == data.iloc[0]).all()].columns.values)
    data = data.loc[:, (data != data.iloc[0]).any()]  # Remove constant columns
    print("---")

    print("Participants: (Rows / 5)")
    print(len(data.index)/5)
    print("---")
    print("Features:")
    print(len(data.columns))
    print("---")
    print("Classes:")
    print(data_labels.value_counts())
    print("---")

    if settingsIn.plotAllHists:
        Plotting.PlotAllFeatures(data, data_labels, settingsIn)

    simplify = False
    if simplify:
        for x in range(data_labels.size):
            if data_labels[x] == 'Assault' or data_labels[x] == 'Journey':
                data_labels[x] = 'Other'

    return data.copy(), data_labels


def feature_selection(data_features, data_labels, usePCA, useFeatSel, dimensionalitySel, dimensionalityPCA):

    # Feature selection

    if usePCA:
        # Feature extraction (PCA)
        pca_features = PCA(n_components=dimensionalityPCA).fit_transform(data_features)
        pca_Data = PCA(n_components=dimensionalityPCA)
        pca_Data.fit(data_features)

        # print("Components: " + str(pca_Data.components_))
        print("PCA:")
        print("Explained variance: " + str(pca_Data.explained_variance_))
        print("Explained variance ratio: " + str(pca_Data.explained_variance_ratio_))
        print(pca_Data)
        print("---")

        data_features = pd.DataFrame(pca_features)
        if useFeatSel:
            selector = SelectKBest(k=dimensionalitySel)
            selector.fit(pca_features, data_labels)
            cols = selector.get_support(indices=True)
            data_features = pd.DataFrame(pca_features[:, cols])
    else:
        if useFeatSel:
            # Feature selection
            selector = SelectKBest(f_classif, k=dimensionalitySel)  # .fit_transform(data_features, data_labels)
            selector.fit(data_features, data_labels)
            cols = selector.get_support(indices=True)
            data_features = data_features.iloc[:, cols]

    return data_features


def normalize(data_features):
    stdScaler = StandardScaler().fit_transform(data_features)
    for y in range(data_features.iloc[0, :].size):
        for x in range(data_features.iloc[:, 0].size):
            data_features.iloc[x, y] = stdScaler[x, y]
    # min_max_scaler = preprocessing.MinMaxScaler()
    # data_features_norm = min_max_scaler.fit_transform(data_features)
    # for y in range(data_features.iloc[0, :].size):
    #    for x in range(data_features.iloc[:, 0].size):
    #        data_features.iloc[x, y] = data_features_norm[x, y]
    return data_features


def handle_invalid_data(settingsIn, data):
    if settingsIn.removeOther:
        data: pd.DataFrame = data[data[settingsIn.classifier_target.value] != 'Other']

    return data


def validate_classification_model(model, X_test, y_test, printStuff):
    y_true = y_test
    y_predict = model.predict(X_test)

    if printStuff:
        print("---")
        print(model)
        print(model.score(X_test, y_test))
        print(balanced_accuracy_score(y_true, y_predict))
        print(confusion_matrix(y_true, y_predict))

    return balanced_accuracy_score(y_true, y_predict), model.score(X_test, y_test), y_true, y_predict


def validate_regression_model(model, X_test, y_test, printStuff):
    y_predict = model.predict(X_test)

    if printStuff:
        print(model.score(X_test, y_test))

    return y_predict
