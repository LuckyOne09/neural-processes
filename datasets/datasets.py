import numpy as np
import torch
import os
from math import pi
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# from model.NeuralProcessModel import NeuralProcess
# from trainer import NeuralProcessTrainer


class FaceFeatureData(Dataset):
    """
    Dataset of face feature vectors extracted from FG-NET by FaceNet dimension reduction.
    size of pair (x,y): size of x is 2048, size of y is 1

    Parameters
    ----------

    featureVectors:
        each of featureVectors is (x,y)
        x.size():
        data from extracted feature vector pre-processed by FaceNet

    num_samples: num of people in dataset

    num_points: num_of_images of each people

    index: select correspoinding data for computing specific people's trained model

    """

    def __init__(self,num_of_people=82,num_of_images=18,index = None):
        self.num_samples = num_of_people
        self.num_points = num_of_images
        self.x_dim = 2048  # x and y dim are fixed for this dataset.
        self.y_dim = 1

        self.indexing = index
        self.featureVectors = []
        filePath = r'./datasets/FeatureVector'
        csvs = os.listdir(filePath)
        FeatureCSVs = map(lambda x: os.path.join(filePath, x), csvs)
        # featureVectors size():
        # (#people(82),#image of each person(18),#features in each image(2048))
        for idx, root_dir in enumerate(FeatureCSVs):
            ageVecotr, featureVector = readFromCSV(root_dir)
            ageTensor = torch.FloatTensor(ageVecotr).unsqueeze(1)
            featureTensor = torch.FloatTensor(featureVector)
            if index is not None:
                if idx == index:
                    self.featureVectors.append((featureTensor,ageTensor))
            else:
                self.featureVectors.append((featureTensor, ageTensor))

    def __getitem__(self, index):
        return self.featureVectors[index]

    def __len__(self):
        #self.numsample == len(self.featureVectors)
        if self.indexing is not None:
            return 1
        return self.num_samples

class FaceFeatureTestData(Dataset):
    """
    Dataset of face feature vectors seperated from FG-NET by FaceNet dimension reduction for testint
    size of pair (x,y): size of x is 2048, size of y is 1
    """

    def __init__(self,testFilePath = r'D:\PycharmProjects\ANP\neural-processes\datasets\TestFeatureVector'):

        self.x_dim = 2048  # x and y dim are fixed for this dataset.
        self.y_dim = 1

        csvs = os.listdir(testFilePath)
        FeatureCSVs = map(lambda x: os.path.join(testFilePath, x), csvs)
        self.testVectors = []
        for root_dir in FeatureCSVs:
            ageVecotr, featureVector = readFromCSV(root_dir)
            ageTensor = torch.FloatTensor(ageVecotr).unsqueeze(1)
            featureTensor = torch.FloatTensor(featureVector)
            self.testVectors.append((featureTensor, ageTensor))

    def __getitem__(self, index):
        return self.testVectors[index]

    def __len__(self):
        #self.numsample == len(self.featureVectors)
        return len(self.testVectors)

class SineData(Dataset):
    """
    Dataset of functions f(x) = a * sin(x - b) where a and b are randomly
    sampled. The function is evaluated from -pi to pi.

    Parameters
    ----------
    amplitude_range : tuple of float
        Defines the range from which the amplitude (i.e. a) of the sine function
        is sampled.

    shift_range : tuple of float
        Defines the range from which the shift (i.e. b) of the sine function is
        sampled.

    num_samples : int
        Number of samples of the function contained in dataset.

    num_points : int
        Number of points at which to evaluate f(x) for x in [-pi, pi].
    """
    def __init__(self, amplitude_range=(-1., 1.), shift_range=(-.5, .5),
                 num_samples=1000, num_points=100):
        self.amplitude_range = amplitude_range
        self.shift_range = shift_range
        self.num_samples = num_samples
        self.num_points = num_points
        self.x_dim = 1  # x and y dim are fixed for this dataset.
        self.y_dim = 1

        # Generate data
        self.data = []
        a_min, a_max = amplitude_range
        b_min, b_max = shift_range
        for i in range(num_samples):
            # Sample random amplitude
            a = (a_max - a_min) * np.random.rand() + a_min
            # Sample random shift
            b = (b_max - b_min) * np.random.rand() + b_min
            # Shape (num_points, x_dim)
            x = torch.linspace(-pi, pi, num_points).unsqueeze(1)
            # Shape (num_points, y_dim)
            y = a * torch.sin(x - b)
            self.data.append((x, y))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.num_samples

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

# def ConstructInputToMergeNet(num_of_test_images,testData_loader):
#     x_dim = 2048
#     y_dim = 1
#     r_dim = 50  # Dimension of representation of context points
#     z_dim = 50  # Dimension of sampled latent variable
#     h_dim = 50  #
#     num_of_people = 3
#     num_of_images = 18
#     batch_size = 1
#     num_context = 17
#     num_target = 1
#     dataset = FaceFeatureData(num_of_people=num_of_people,num_of_images=num_of_images)
#     data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#     for batch in data_loader:
#         break
#     # Use batch to create random set of context points
#     x, y = batch
#     x_context, y_context, _, _ = NeuralProcessTrainer.context_target_split(x[0:1], y[0:1],
#                                                       num_context,
#                                                       num_target)
#
#     modelPath = r'D:\PycharmProjects\ANP\neural-processes\trained_models\age_estimation\smallTrained'
#     models = os.listdir(modelPath)
#     smallModels = map(lambda x: os.path.join(modelPath, x), models)
#
#
#     test_target = 0
#     resultsOnPretrainedModelsList = []
#     for idx, root_dir in enumerate(smallModels):
#         #load model
#         testNeuralprocess = NeuralProcess(x_dim, y_dim, r_dim, z_dim, h_dim)
#         testModelPath = r'D:\PycharmProjects\ANP\neural-processes\trained_models\age_estimation\smallTrained\smallTrained' + str(idx) + r'.ckpt'
#         testNeuralprocess.load_state_dict(torch.load(testModelPath))
#         testNeuralprocess.training = False
#
#         resultsOnPretrainedModel = []
#         for x_target, y_target in testData_loader:
#             test_target = y_target
#             avg_mu = 0
#             for i in range(10):
#                 p_y_pred = testNeuralprocess(x_context, y_context, x_target)
#                 # Extract mean of distribution
#                 mu = p_y_pred.loc.detach()
#                 avg_mu += mu
#             avg_mu = avg_mu / 10
#             avg_mu = avg_mu.view(18)
#             resultsOnPretrainedModel.append(avg_mu.tolist())
#         resultsOnPretrainedModelsList.append(resultsOnPretrainedModel)
#
#     resultsOnPretrainedModels = []
#     for i in range(num_of_test_images):
#         resultsWithSinglePerson = []
#         for list in resultsOnPretrainedModelsList:
#             resultsWithSinglePerson.append(list[0][i])
#         resultsOnPretrainedModels.append(resultsWithSinglePerson)
#     resultsOnPretrainedModels = torch.FloatTensor(resultsOnPretrainedModels)
#     return resultsOnPretrainedModels