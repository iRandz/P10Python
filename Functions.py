import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler


def recalc_manage(row):
    if row['Type'] == 'Manage':
        if row['Assault mean'] > row['Journey mean']:
            return 'Assault'
        return 'Journey'
    return row['Type']


def calc_ratio(row, timeName, interactionName):
    if row[interactionName] == 0:
        return 0
    return row[timeName] / row[interactionName]


def calc_derived_features(dataframe):
    dataframe['MajorLoreTimePr'] = dataframe.apply(
        lambda row: calc_ratio(row, 'MajorLore reading time', 'MajorLore interactions'), axis=1)
    dataframe['LoreTimePr'] = dataframe.apply(
        lambda row: calc_ratio(row, 'Lore reading time', 'Lore interactions'), axis=1)
    # dataframe['MajorLoreRatioClose'] = dataframe.apply (lambda row: CalcRatio(row, 'MajorLore interactions',
    # 'MajorLore seen'), axis=1)
    dataframe['LoreRatioSeen'] = dataframe.apply(
        lambda row: calc_ratio(row, 'Lore interactions', 'Lore seen'), axis=1)
    dataframe['LoreRatioClose'] = dataframe.apply(
        lambda row: calc_ratio(row, 'Lore interactions', 'Lore close'), axis=1)
    dataframe['ResourceRatioSeen'] = dataframe.apply(
        lambda row: calc_ratio(row, 'Resources', 'Resources seen '), axis=1)
    dataframe['ResourceRatioClose'] = dataframe.apply(
        lambda row: calc_ratio(row, 'Resources', 'Resources close'), axis=1)
    dataframe['EnemyRatioSeen'] = dataframe.apply(lambda row: calc_ratio(row, 'Kills', 'Enemies seen'), axis=1)
    dataframe['EnemyRatioClose'] = dataframe.apply(
        lambda row: calc_ratio(row, 'Kills', 'Enemies close'), axis=1)
    dataframe['ResourceRatioSeen'] = dataframe.apply(
        lambda row: calc_ratio(row, 'Resources', 'Resources seen '), axis=1)
    # dataframe['MapTimePr'] = dataframe.apply (lambda row: CalcRatio(row, 'MapTime', 'Opened map'), axis=1)

    return dataframe


def process_data(data, settingsIn):
    recalcManage = settingsIn.recalcManage
    removeManage = settingsIn.removeManage
    if settingsIn.test == settingsIn.CurrentTest.DIMENSIONALITY or settingsIn.test == settingsIn.CurrentTest.CLASSIFICATION:
        target = settingsIn.classifier_target
    elif settingsIn.test == settingsIn.CurrentTest.REGRESSION:
        target = settingsIn.regressor_target
    else:
        # Do nothing
        print("What u doing?")
        return

    def safe_pop(data_to_pop_from, popValue):
        if popValue in data_to_pop_from.columns:
            data_to_pop_from.pop(popValue)

    if recalcManage:
        data['Type'] = data.apply(lambda row: recalc_manage(row), axis=1)

    if removeManage:
        data = data[data['Type'] != 'Manage']

    data_labels = data.pop(target.value)

    safe_pop(data, 'Type')
    safe_pop(data, 'Gender')

    # data.pop('Weekly playtime')
    data.pop('Participant ID')
    # data.pop('Age')
    data.pop('Journey mean')
    data.pop('Manage mean')
    data.pop('Assault mean')
    # data.pop('Unique tiles')
    # data.pop('Deaths')
    # data.pop('MajorLore seen')
    # data.pop('MajorLore reading time')
    # data.pop('MajorLore close')
    # data.pop('MajorLore interactions')

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
            selector = SelectKBest(chi2, k=dimensionalitySel)  # .fit_transform(data_features, data_labels)
            selector.fit(data_features, data_labels)
            cols = selector.get_support(indices=True)
            data_features = data_features.iloc[:, cols]

    return data_features


def normalize(data_features):
    stdScaler = StandardScaler().fit_transform(data_features)
    for y in range(data_features.iloc[0, :].size):
        for x in range(data_features.iloc[:, 0].size):
            data_features.iloc[x, y] = stdScaler[x, y]
    min_max_scaler = preprocessing.MinMaxScaler()
    data_features_norm = min_max_scaler.fit_transform(data_features)
    for y in range(data_features.iloc[0, :].size):
        for x in range(data_features.iloc[:, 0].size):
            data_features.iloc[x, y] = data_features_norm[x, y]
    return data_features


def handle_invalid_data(data_labels, removeOther, data):
    for x in range(data_labels.size):
        y = data_labels.iloc[x]
        if y != 'Manage' and y != 'Journey' and y != 'Assault' and y != 'Other':
            data_labels[x] = 'Other'
    if removeOther:
        data = data[data['Type'] != 'Other']

    return data_labels, data


def validate_classification_model(model, X_test, y_test, printStuff):
    y_true = y_test
    y_predict = model.predict(X_test)

    if printStuff:
        print(model.score(X_test, y_test))
        print(balanced_accuracy_score(y_true, y_predict))
        print(confusion_matrix(y_true, y_predict))

    return balanced_accuracy_score(y_true, y_predict)


def validate_regression_model(model, X_test, y_test, printStuff):
    y_true = y_test
    y_predict = model.predict(X_test)

    if printStuff:
        print(model.score(X_test, y_test))

    return y_predict
