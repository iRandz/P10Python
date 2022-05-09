# File for separating data into different categories
import numpy as np

import pandas as pd
import FeatureDict as fD


def ByDay(dataIn):
	data_0: pd.DataFrame = dataIn.loc[dataIn[fD.DAY] == 0]
	data_1 = dataIn.loc[dataIn[fD.DAY] == 1]
	data_2 = dataIn.loc[dataIn[fD.DAY] == 2]
	data_3 = dataIn.loc[dataIn[fD.DAY] == 3]
	data_4 = dataIn.loc[dataIn[fD.DAY] == 4]

	data_0.to_csv('data_0.csv', sep=';', index=False)
	data_1.to_csv('data_1.csv', sep=';', index=False)
	data_2.to_csv('data_2.csv', sep=';', index=False)
	data_3.to_csv('data_3.csv', sep=';', index=False)
	data_4.to_csv('data_4.csv', sep=';', index=False)


def AccumulativeDays(dataIn):
	data_0: pd.DataFrame = dataIn.loc[dataIn[fD.DAY] == 0]
	data_1 = dataIn.loc[dataIn[fD.DAY] == 1]
	data_2 = dataIn.loc[dataIn[fD.DAY] == 2]
	data_3 = dataIn.loc[dataIn[fD.DAY] == 3]
	data_4 = dataIn.loc[dataIn[fD.DAY] == 4]

	IDs = dataIn.copy().pop(fD.ID)

	accuDay1: pd.DataFrame = data_0.copy(True)
	accuDay2: pd.DataFrame = data_0.copy(True)
	accuDay3: pd.DataFrame = data_0.copy(True)
	accuDay4: pd.DataFrame = data_0.copy(True)

	for i, currrentID in enumerate(IDs.unique()):  # Iterate over participant ids
		tempHolder: pd.DataFrame = dataIn.loc[dataIn[fD.ID] == currrentID]
		tempDay1 = tempHolder.iloc[0, :].copy()
		tempDay2 = tempHolder.iloc[0, :].copy()
		tempDay3 = tempHolder.iloc[0, :].copy()
		tempDay4 = tempHolder.iloc[0, :].copy()

		for j in range(tempHolder.iloc[0].size):  # Iterate over features
			if isinstance(tempHolder.iloc[0, j], float) or isinstance(tempHolder.iloc[0, j], np.int64):
				mask = (fD.ID, fD.J_MEAN, fD.M_MEAN, fD.A_MEAN, fD.DAY, fD.AGE, fD.P_PARTICIPANT, fD.PLAYTIME)
				if tempHolder.iloc[:, j].name in mask:
					continue

				if tempHolder.iloc[:, j].name is fD.DIST:
					tempDay1.iloc[j] = np.max(tempHolder.iloc[0, j], tempHolder.iloc[1, j])
					tempDay2.iloc[j] = np.max(tempDay1.iloc[j], tempHolder.iloc[2, j])
					tempDay3.iloc[j] = np.max(tempDay2.iloc[j], tempHolder.iloc[3, j])
					tempDay4.iloc[j] = np.max(tempDay3.iloc[j], tempHolder.iloc[4, j])

				tempDay1.iloc[j] = tempHolder.iloc[0, j] + tempHolder.iloc[1, j]
				tempDay2.iloc[j] = tempDay1.iloc[j] + tempHolder.iloc[2, j]
				tempDay3.iloc[j] = tempDay2.iloc[j] + tempHolder.iloc[3, j]
				tempDay4.iloc[j] = tempDay3.iloc[j] + tempHolder.iloc[4, j]

		accuDay1.iloc[i, :] = tempDay1
		accuDay2.iloc[i, :] = tempDay2
		accuDay3.iloc[i, :] = tempDay3
		accuDay4.iloc[i, :] = tempDay4

	accuDay1.to_csv('AccumulativeDays1.csv', sep=';', index=False)
	accuDay2.to_csv('AccumulativeDays2.csv', sep=';', index=False)
	accuDay3.to_csv('AccumulativeDays3.csv', sep=';', index=False)
	accuDay4.to_csv('AccumulativeDays4.csv', sep=';', index=False)


def ByPlaytime(dataIn, valueSplit):
	data_lowPlaytime: pd.DataFrame = dataIn.loc[dataIn[fD.PLAYTIME] <= valueSplit]
	data_highPlaytime = dataIn.loc[dataIn[fD.PLAYTIME] > valueSplit]

	data_lowPlaytime.to_csv('data_lowPlayTime.csv', sep=';', index=False)
	data_highPlaytime.to_csv('data_highPlayTime.csv', sep=';', index=False)


def ByMeans(dataIn, threshold):
	data_ManageDif: pd.DataFrame = dataIn.loc[dataIn[fD.M_MEAN] > dataIn[fD.A_MEAN] + threshold]
	data_ManageDif = data_ManageDif.loc[data_ManageDif[fD.M_MEAN] > data_ManageDif[fD.J_MEAN] + threshold]

	data_JourneyDif = dataIn.loc[dataIn[fD.J_MEAN] > dataIn[fD.A_MEAN] + threshold]
	data_JourneyDif = data_JourneyDif.loc[data_JourneyDif[fD.J_MEAN] > data_JourneyDif[fD.M_MEAN] + threshold]

	data_AssaultDif = dataIn.loc[dataIn[fD.A_MEAN] > dataIn[fD.M_MEAN] + threshold]
	data_AssaultDif = data_AssaultDif.loc[data_AssaultDif[fD.A_MEAN] > data_AssaultDif[fD.J_MEAN] + threshold]

	# data_lowMeanDif: pd.DataFrame = dataIn.loc[dataIn[fD.M_MEAN] > dataIn[fD.A_MEAN]]
	data_highMeanDif = pd.concat([data_AssaultDif, data_JourneyDif, data_ManageDif])

	data_highMeanDif.to_csv('data_highMeanDif.csv', sep=';', index=False)


def ByPlayedBefore(dataIn):
	data_neverPlayedBefore: pd.DataFrame = dataIn.loc[dataIn[fD.P_PARTICIPANT] < 1]

	data_neverPlayedBefore.to_csv('data_neverPlayedBefore.csv', sep=';', index=False)


data = pd.read_csv("Data/CombinedDayLog.CSV", sep=';')

# ByDay(data)
# ByPlaytime(data, 3)
# ByMeans(data, 1)
# ByPlayedBefore(data)
AccumulativeDays(data)
