import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import balanced_accuracy_score, confusion_matrix


def RecalcManage(row):
    if row['Type'] == 'Manage':
        if row['Assault mean'] > row['Journey mean']:
            return 'Assault'
        return 'Journey'
    return row['Type']


def CalcRatio(row, timeName, interactionName):
    if row[interactionName] == 0:
        return 0
    return row[timeName] / row[interactionName]


def CalcDerivedFeatures(dataframe):
    dataframe['MajorLoreTimePr'] = dataframe.apply(
        lambda row: CalcRatio(row, 'MajorLore reading time', 'MajorLore interactions'), axis=1)
    dataframe['LoreTimePr'] = dataframe.apply(
        lambda row: CalcRatio(row, 'Lore reading time', 'Lore interactions'), axis=1)
    # dataframe['MajorLoreRatioClose'] = dataframe.apply (lambda row: CalcRatio(row, 'MajorLore interactions', 'MajorLore seen'), axis=1)
    dataframe['LoreRatioSeen'] = dataframe.apply(
        lambda row: CalcRatio(row, 'Lore interactions', 'Lore seen'), axis=1)
    dataframe['LoreRatioClose'] = dataframe.apply(
        lambda row: CalcRatio(row, 'Lore interactions', 'Lore close'), axis=1)
    dataframe['ResourceRatioSeen'] = dataframe.apply(
        lambda row: CalcRatio(row, 'Resources', 'Resources seen '), axis=1)
    dataframe['ResourceRatioClose'] = dataframe.apply(
        lambda row: CalcRatio(row, 'Resources', 'Resources close'), axis=1)
    dataframe['EnemyRatioSeen'] = dataframe.apply(lambda row: CalcRatio(row, 'Kills', 'Enemies seen'), axis=1)
    dataframe['EnemyRatioClose'] = dataframe.apply(
        lambda row: CalcRatio(row, 'Kills', 'Enemies close'), axis=1)
    dataframe['ResourceRatioSeen'] = dataframe.apply(
        lambda row: CalcRatio(row, 'Resources', 'Resources seen '), axis=1)
    # dataframe['MapTimePr'] = dataframe.apply (lambda row: CalcRatio(row, 'MapTime', 'Opened map'), axis=1)

    return dataframe


def ProcessData(data, recalcManage, removeManage, target):
    def SafePop(data, popValue):
        if popValue in data.columns:
            data.pop(popValue)

    if recalcManage:
        data['Type'] = data.apply(lambda row: RecalcManage(row), axis=1)

    if removeManage:
        data = data[data['Type'] != 'Manage']

    data_labels = data.pop(target.value)

    SafePop(data, 'Type')
    SafePop(data, 'Gender')

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


def FeatureSelection(data_features, data_labels, usePCA, useFeatSel, dimensionalitySel, dimensionalityPCA):

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


def Normalize(data_features):
    min_max_scaler = preprocessing.MinMaxScaler()
    data_features_norm = min_max_scaler.fit_transform(data_features)
    for y in range(data_features.iloc[0, :].size):
        for x in range(data_features.iloc[:, 0].size):
            data_features.iloc[x, y] = data_features_norm[x, y]
    return data_features


def HandleInvalidData(data_labels, removeOther, data):
    for x in range(data_labels.size):
        y = data_labels.iloc[x]
        if y != 'Manage' and y != 'Journey' and y != 'Assault' and y != 'Other':
            data_labels[x] = 'Other'
    if removeOther:
        data = data[data['Type'] != 'Other']

    return data_labels, data


def ValidateModel(model, X_test, y_test, printStuff):
    y_true = y_test
    y_pred = model.predict(X_test)

    if printStuff:
        print(model.score(X_test, y_test))
        print(balanced_accuracy_score(y_true, y_pred))
        print(confusion_matrix(y_true, y_pred))

    return balanced_accuracy_score(y_true, y_pred)
