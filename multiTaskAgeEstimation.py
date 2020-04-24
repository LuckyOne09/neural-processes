import torch
from torch.utils.data import DataLoader
from datasets import FaceFeatureData, FaceFeatureTestData
from neural_process import NeuralProcess
from training import NeuralProcessTrainer
from utils import context_target_split
import os
from mergeNet import MergeNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x_dim = 2048
y_dim = 1
r_dim = 50  # Dimension of representation of context points
z_dim = 50  # Dimension of sampled latent variable
h_dim = 50  #
num_of_people = 3
num_of_images=18
batch_size = 1
num_context = 17
num_target = 1


def wholeDatasetPretrain():
    dataset = FaceFeatureData(num_of_people=num_of_people,num_of_images=num_of_images)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    neuralprocess = NeuralProcess(x_dim, y_dim, r_dim, z_dim, h_dim)
    optimizer = torch.optim.Adam(neuralprocess.parameters(), lr=3e-4)
    np_trainer = NeuralProcessTrainer(device, neuralprocess, optimizer,
                                      num_context_range=(num_context, num_context),
                                      num_extra_target_range=(num_target, num_target),
                                      print_freq=200)
    neuralprocess.training = True
    np_trainer.train(data_loader, 500)
    #save first model parameters trained on the whole dataset
    torch.save(neuralprocess.state_dict(), r'D:\PycharmProjects\ANP\neural-processes-oxford\trained_models\age_estimation\firstWholeTrained.ckpt')

def trainOnEachPerson():
    singlePersonDatasets = []
    for idx in range(num_of_people):
        singlePersonDataset = FaceFeatureData(num_of_people=num_of_people, num_of_images=num_of_images, index=idx)
        singlePersonDatasets.append(singlePersonDataset)

    for idx, singlePersonDataset in enumerate(singlePersonDatasets):
        # load model
        smallNeuralprocess = NeuralProcess(x_dim, y_dim, r_dim, z_dim, h_dim)
        smallNeuralprocess.load_state_dict(torch.load(
            r'D:\PycharmProjects\ANP\neural-processes-oxford\trained_models\age_estimation\firstWholeTrained.ckpt'))

        singleData_loader = DataLoader(singlePersonDataset, batch_size=batch_size, shuffle=True)
        smallOptimizer = torch.optim.SGD(smallNeuralprocess.parameters(), lr=3e-5)
        smallNp_trainer = NeuralProcessTrainer(device, smallNeuralprocess, smallOptimizer,
                                               num_context_range=(num_context, num_context),
                                               num_extra_target_range=(num_target, num_target),
                                               print_freq=200)
        smallNeuralprocess.training = True
        smallNp_trainer.train(singleData_loader, 50)
        # save first model parameters trained on the whole dataset
        path = r'D:\PycharmProjects\ANP\neural-processes-oxford\trained_models\age_estimation\smallTrained\smallTrained' + str(
            idx) + r'.ckpt'
        torch.save(smallNeuralprocess.state_dict(), path)


def ConstructInputToMergeNet(num_of_test_images,testData_loader):
    dataset = FaceFeatureData(num_of_people=num_of_people,num_of_images=num_of_images)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for batch in data_loader:
        break
    # Use batch to create random set of context points
    x, y = batch
    x_context, y_context, _, _ = context_target_split(x[0:1], y[0:1],
                                                      num_context,
                                                      num_target)

    modelPath = r'D:\PycharmProjects\ANP\neural-processes-oxford\trained_models\age_estimation\smallTrained'
    models = os.listdir(modelPath)
    smallModels = map(lambda x: os.path.join(modelPath, x), models)


    test_target = 0
    resultsOnPretrainedModelsList = []
    for idx, root_dir in enumerate(smallModels):
        #load model
        testNeuralprocess = NeuralProcess(x_dim, y_dim, r_dim, z_dim, h_dim)
        testModelPath = r'D:\PycharmProjects\ANP\neural-processes-oxford\trained_models\age_estimation\smallTrained\smallTrained' + str(idx) + r'.ckpt'
        testNeuralprocess.load_state_dict(torch.load(testModelPath))
        testNeuralprocess.training = False

        resultsOnPretrainedModel = []
        for x_target, y_target in testData_loader:
            test_target = y_target
            avg_mu = 0
            for i in range(10):
                p_y_pred = testNeuralprocess(x_context, y_context, x_target)
                # Extract mean of distribution
                mu = p_y_pred.loc.detach()
                avg_mu += mu
            avg_mu = avg_mu / 10
            print('avg_mu.size(): ',avg_mu.size())
            print(avg_mu)
            avg_mu = avg_mu.view(18)
            print('after view operation')
            print('avg_mu.size(): ',avg_mu.size())
            print(avg_mu)
            resultsOnPretrainedModel.append(avg_mu.tolist())
        resultsOnPretrainedModelsList.append(resultsOnPretrainedModel)
    print(resultsOnPretrainedModelsList)

    resultsOnPretrainedModels = []
    for i in range(num_of_test_images):
        resultsWithSinglePerson = []
        for list in resultsOnPretrainedModelsList:
            resultsWithSinglePerson.append(list[0][i])
        resultsOnPretrainedModels.append(resultsWithSinglePerson)
    resultsOnPretrainedModels = torch.FloatTensor(resultsOnPretrainedModels)
    return resultsOnPretrainedModels


def trainingMergeNet(num_of_test_images,resultsOnPretrainedModels,test_target):
    mergeNet = MergeNet(number_of_trained_people=3)
    mergeOptimizer = torch.optim.Adam(mergeNet.parameters(), lr=3e-3)
    mergeEpoch = 350
    criterion = torch.nn.MSELoss()
    test_target = test_target.view(num_of_test_images,1)
    merge_loss_history = []
    for epoch in range(mergeEpoch):
        mergeOptimizer.zero_grad()
        mergeResult = mergeNet(resultsOnPretrainedModels)
        loss = criterion(mergeResult, test_target)
        loss.backward()
        mergeOptimizer.step()
        print("Epoch: {}, loss: {}".format(epoch, loss))
        merge_loss_history.append(loss)
    #save mergeNet
    path = r'D:\PycharmProjects\ANP\neural-processes-oxford\trained_models\mergeNet.ckpt'
    torch.save(mergeNet.state_dict(),path)
