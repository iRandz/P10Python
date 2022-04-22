# File for moving the map images to the correct folder
import shutil
import pandas as pd
import FeatureDict as fD


def MoveExObSplit(dataIn, type):
    if (type == "Objective"):
        obData: pd.DataFrame = dataIn.loc[dataIn[fD.OBJ_EXP] == "Objective"]
        for i in range(obData.iloc[:, 0].size):
            rowInfo = obData.iloc[i]
            idString = rowInfo[fD.ID]
            dayString = rowInfo[fD.DAY]
            if(idString == 93):
                continue
            print(str(idString) + "_" + str(dayString))
            shutil.move("MapImages/" + str(idString) + "_" + str(dayString) + ".png", "ExObImages/Objective/Objective" +
                        str(idString) + "_" + str(dayString) + ".png")

    if (type == "Exploration"):
        exData: pd.DataFrame = dataIn.loc[dataIn[fD.OBJ_EXP] == "Exploration"]
        for i in range(exData.iloc[:, 0].size):
            rowInfo = exData.iloc[i]
            idString = rowInfo[fD.ID]
            dayString = rowInfo[fD.DAY]
            if(idString == 93):
                continue
            print(str(idString) + "_" + str(dayString))
            shutil.move("MapImages/" + str(idString) + "_" + str(dayString) + ".png", "ExObImages/Exploration/Exploration" + str(idString) + "_"
                        + str(dayString) + ".png")



def JMASplit(dataIn):
    jouData: pd.DataFrame = dataIn.loc[dataIn[fD.TYPE] == "Journey"]
    manData: pd.DataFrame = dataIn.loc[dataIn[fD.TYPE] == "Manage"]
    assData: pd.DataFrame = dataIn.loc[dataIn[fD.TYPE] == "Assault"]

    for i in range(jouData.iloc[:, 0].size):
        rowInfo = jouData.iloc[i]
        idString = rowInfo[fD.ID]
        dayString = rowInfo[fD.DAY]
        if(idString == 93):
            continue
        print("Journey " + str(idString) + "_" + str(dayString))
        shutil.move("MapImages/" + str(idString) + "_" + str(dayString) + ".png", "PlayerTypesImages/Journey/Journey" +
                    str(idString) + "_" + str(dayString) + ".png")

    for i in range(manData.iloc[:, 0].size):
        rowInfo = manData.iloc[i]
        idString = rowInfo[fD.ID]
        dayString = rowInfo[fD.DAY]
        if(idString == 93):
            continue
        print("Manage " + str(idString) + "_" + str(dayString))
        shutil.move("MapImages/" + str(idString) + "_" + str(dayString) + ".png", "PlayerTypesImages/Manage/Manage" +
                    str(idString) + "_" + str(dayString) + ".png")

    for i in range(assData.iloc[:, 0].size):
        rowInfo = assData.iloc[i]
        idString = rowInfo[fD.ID]
        dayString = rowInfo[fD.DAY]
        if(idString == 93):
            continue
        print("Assault " + str(idString) + "_" + str(dayString))
        shutil.move("MapImages/" + str(idString) + "_" + str(dayString) + ".png", "PlayerTypesImages/Assault/Assault" +
                    str(idString) + "_" + str(dayString) + ".png")



data = pd.read_csv("CombinedDayLog.csv", sep=';')

#MoveExObSplit(data, "Objective")
#JMASplit(data)