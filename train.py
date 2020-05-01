import os

import matplotlib.pyplot as plt
import torch

from datasets.datasets import FaceFeatureData, FaceFeatureTestData, ConstructInputToMergeNet
from data_loader.data_loader import FGNetDataLoader
from model.NeuralProcessModel import NeuralProcess
from model.models import MergeNet
from trainer.NP_trainer import NeuralProcessTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_of_people = 3
num_of_images=18
x_dim = 2048
y_dim = 1
r_dim = 50  # Dimension of representation of context points
z_dim = 50  # Dimension of sampled latent variable
h_dim = 50  # Dimension of hidden layers in encoder and decoder

batch_size = 1
num_context = 17
num_target = 1
# Create dataset
dataset = FaceFeatureData(num_of_people=num_of_people,num_of_images=num_of_images)
data_loader = FGNetDataLoader(dataset, batch_size=batch_size, shuffle=True)

neuralprocess = NeuralProcess(x_dim, y_dim, r_dim, z_dim, h_dim)
optimizer = torch.optim.Adam(neuralprocess.parameters(), lr=3e-4)

np_trainer = NeuralProcessTrainer(device, neuralprocess, optimizer,
                                  num_context_range=(num_context, num_context),
                                  num_extra_target_range=(num_target, num_target),
                                  data_loader=data_loader)
np_trainer.train(100)
torch.save(neuralprocess.state_dict(), r'.\trained_models\age_estimation\firstWholeTrained.ckpt')

singlePersonDatasets = []
for idx in range(num_of_people):
    singlePersonDataset = FaceFeatureData(num_of_people=num_of_people,num_of_images=num_of_images,index=idx)
    singlePersonDatasets.append(singlePersonDataset)
    for idx, singlePersonDataset in enumerate(singlePersonDatasets):
        # load model

        smallNeuralprocess = NeuralProcess(x_dim, y_dim, r_dim, z_dim, h_dim)
        smallNeuralprocess.load_state_dict(torch.load(
            r'.\trained_models\age_estimation\firstWholeTrained.ckpt'))
        # Freeze the encoder part of each model
        for child in smallNeuralprocess.children():
            for param in child.parameters():
                param.requires_grad = False
            break
        singleData_loader = FGNetDataLoader(singlePersonDataset, batch_size=batch_size, shuffle=True)
        smallOptimizer = torch.optim.Adam(smallNeuralprocess.parameters(), lr=3e-5)
        smallNp_trainer = NeuralProcessTrainer(device, smallNeuralprocess, smallOptimizer,
                                               num_context_range=(num_context, num_context),
                                               num_extra_target_range=(num_target, num_target),
                                               data_loader=singleData_loader)
        smallNp_trainer.train(100)
        path = r'.\trained_models\age_estimation\smallTrained\smallTrained' + str(idx) + r'.ckpt'
        torch.save(smallNeuralprocess.state_dict(), path)

for batch in data_loader:
    break
# Use batch to create random set of context points
x, y = batch
x_context, y_context, _, _ =  NeuralProcessTrainer.context_target_split(x[0:1], y[0:1],
                                                  num_context,
                                                  num_target)


modelPath = r'.\trained_models\age_estimation\smallTrained'
models = os.listdir(modelPath)
smallModels = map(lambda x: os.path.join(modelPath, x), models)

num_of_test_images = 18
test_target = 0
resultsOnPretrainedModelsList = []
for idx, root_dir in enumerate(smallModels):
    #load model
    testNeuralprocess = NeuralProcess(x_dim, y_dim, r_dim, z_dim, h_dim)
    testModelPath = r'.\trained_models\age_estimation\smallTrained\smallTrained' + str(idx) + r'.ckpt'
    testNeuralprocess.load_state_dict(torch.load(testModelPath))
    testNeuralprocess.training = False

    testDataset = FaceFeatureTestData()
    testData_loader = FGNetDataLoader(testDataset, batch_size=batch_size, shuffle=True)
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
        avg_mu = avg_mu.view(18)
        resultsOnPretrainedModel.append(avg_mu.tolist())
    resultsOnPretrainedModelsList.append(resultsOnPretrainedModel)

resultsOnPretrainedModels = []
for i in range(num_of_test_images):
    resultsWithSinglePerson = []
    for list in resultsOnPretrainedModelsList:
        resultsWithSinglePerson.append(list[0][i])
    resultsOnPretrainedModels.append(resultsWithSinglePerson)
resultsOnPretrainedModels = torch.FloatTensor(resultsOnPretrainedModels)


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
plt.plot(range(len(merge_loss_history)),merge_loss_history)
plt.show()
#save mergeNet
path = r'.\trained_models\mergeNet.ckpt'
torch.save(mergeNet.state_dict(),path)


finalTestDataset = FaceFeatureTestData(r'.\datasets\FinalTestFeatureVector')
finalTestData_loader = FGNetDataLoader(finalTestDataset, batch_size=batch_size, shuffle=True)
finalResultsOnPretrainedModels = ConstructInputToMergeNet(num_of_test_images,finalTestData_loader)
testMergeNet = MergeNet(number_of_trained_people=3)
testMergeNet.load_state_dict(torch.load(r'.\trained_models\mergeNet.ckpt'))

final_predict_value = testMergeNet(finalResultsOnPretrainedModels)
print('final result: ')
print(final_predict_value)
for _,target_y in finalTestData_loader:
    print(target_y)