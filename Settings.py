from enum import Enum


class Settings:
    class ClassTarget(Enum):
        TYPE = 'Type'
        GENDER = 'Gender'
        OBJOREXP = 'Obj or exp'
        DAYS = 'Day'

    class RegressionTarget(Enum):
        PLAYTIME = 'Weekly playtime'
        AGE = 'Age'
        SPAM = 'Shots fired'
        KILLS = 'Kills'
        RESOURCES = 'Resources'
        LORE = 'Lore interactions'

    class CurrentTest(Enum):
        DIMENSIONALITY = 0
        CLASSIFICATION = 1
        REGRESSION = 2
        ROC = 3

    dataFile = "CombinedDays.csv"
    test = CurrentTest.DIMENSIONALITY
    plotAllHists = 0
    classifier_target = ClassTarget.OBJOREXP
    regressor_target = RegressionTarget.PLAYTIME

    show2D = 1
    show3D = 0
    column1X = 0
    column2Y = 1
    column3Z = 2

    recalcManage = 0
    recalcJourney = 0
    removeManage = 0
    removeOther = 1
    normalize = 1

    usePCA = 0
    useFeatSel = 1

    dimensionalityPCA = 3
    dimensionalitySel = 7
    groups = None
