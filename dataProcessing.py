import os
import pandas as pd
import torch


def toFeatureVector(x,row_num):
    x_elements = x[2:len(x)-2]
    elements = x_elements.split(' ')
    features = []
    for idx, e in enumerate(elements):
        if (len(e) > 0):
            if (e[len(e) - 1] == '\n'):
                e = e[0:len(e) - 1]
            try:
                features.append(float(e))
            except Exception as ex:
                print(idx)
                print('row_num: ',row_num)
                print(ex)
    return features

def readFromCSV(root_dir):
    FaceFeatureData = pd.read_csv(root_dir, names=['age', 'featureVector'])
    ageVector = []
    featureVector = []
    for i in range(FaceFeatureData.shape[0]):
        age = FaceFeatureData.iloc[i,0]
        x = FaceFeatureData.iloc[i, 1]
        ageVector.append(age)
        featureVector.append(toFeatureVector(x,i))
    return ageVector, featureVector

#get FGnet data(Separate people in different list)
def getData():
    filePath = './FeatureVector/'
    csvs = os.listdir(filePath)
    FeatureCSVs = map(lambda x: os.path.join(filePath,x),csvs)
    #featureVectors size():
    # (#people(82),#image of each person(18),#features in each image(2048))
    ageVecotrs = []
    featureVectors = []
    for root_dir in FeatureCSVs:
        ageVecotr, featureVector = readFromCSV(root_dir)
        ageVecotrs.append(ageVecotr)
        featureVectors.append(featureVector)
    return ageVecotrs, featureVectors

def getTestData():
    testFilePath = r'D:\PycharmProjects\ANP\neural-processes-oxford\FeatureVector\test'
    csvs = os.listdir(testFilePath)
    FeatureCSVs = map(lambda x: os.path.join(testFilePath, x), csvs)
    testVectors = []
    for root_dir in FeatureCSVs:
        ageVecotr, featureVector = readFromCSV(root_dir)
        ageTensor = torch.FloatTensor(ageVecotr).unsqueeze(1)
        featureTensor = torch.FloatTensor(featureVector)
        testVectors.append((featureTensor, ageTensor))

    return testVectors

# getData()
