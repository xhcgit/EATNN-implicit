import pickle as pk
import scipy.sparse as sp
import numpy as np
import os


def loadData(datasetStr, cv):
    if datasetStr == "Tianchi_time":
        return loadData2(datasetStr, cv)
    DIR = os.path.join(os.path.dirname(os.getcwd()), "dataset", datasetStr, 'implicit', "cv{0}".format(cv))
    print(os.getcwd())
    print(os.path.dirname(os.getcwd()))

    TRAIN_FILE = DIR + '/train.csv'
    TEST_FILE  = DIR + '/test.csv'
    TEST_DATA_FILE  = DIR + '/test_data.csv'
    TRAIN_FILE_TIME = DIR + '/train_time.csv'
    TRUST_FILE = DIR + '/trust.csv'
    print(TRAIN_FILE)
    with open(TRAIN_FILE, 'rb') as fs:
        trainMat = pk.load(fs)
    with open(TEST_DATA_FILE, 'rb') as fs:
        testData = pk.load(fs)
    with open(TRUST_FILE, 'rb') as fs:
        trustMat = pk.load(fs)
    return trainMat, testData, trustMat

def loadData2(datasetStr, cv):
    assert datasetStr == "Tianchi_time"
    DIR = os.path.join(os.path.dirname(os.getcwd()), "dataset", datasetStr, 'implicit', "cv{0}".format(cv))
    with open(DIR + '/pvTime.csv'.format(cv), 'rb') as fs:
        pvTimeMat = pk.load(fs)
    with open(DIR + '/cartTime.csv'.format(cv), 'rb') as fs:
        cartTimeMat = pk.load(fs)
    with open(DIR + '/favTime.csv'.format(cv), 'rb') as fs:
        favTimeMat = pk.load(fs)
    with open(DIR + '/buyTime.csv'.format(cv), 'rb') as fs:
        buyTimeMat = pk.load(fs)
    with open(DIR + "/test_data.csv".format(cv), 'rb') as fs:
        test_data = pk.load(fs)
    with open(DIR + "/valid_data.csv".format(cv), 'rb') as fs:
        valid_data = pk.load(fs)
    
    with open(DIR + "/trust.csv", 'rb') as fs:
        trust = pk.load(fs)
    interatctMat = ((pvTimeMat + cartTimeMat + favTimeMat + buyTimeMat) != 0) * 1
    return interatctMat, test_data, trust