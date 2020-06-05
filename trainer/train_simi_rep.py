import os

import torch
import numpy as np
import matplotlib.pyplot as plt
from data_loader.data_loader import FGNetDataLoader
from datasets.datasets import FaceFeatureData
from model.NeuralProcessModel import NeuralProcess
from model.models import Simi_repesentation, MergeNet
from trainer import NeuralProcessTrainer

x_dim = 2048
y_dim = 1
r_dim = 50  # Dimension of representation of context points
z_dim = 50  # Dimension of sampled latent variable
h_dim = 50  # Dimension of hidden layers in encoder and decoder

batch_size = 1
num_of_people = 3
num_of_images = 18
singlePersonDatasets = []

for idx in range(num_of_people):
    singlePersonDataset = FaceFeatureData(num_of_people=num_of_people,num_of_images=num_of_images,index=idx)
    singlePersonDatasets.append(singlePersonDataset)

"""
get the original representation from encoder of general NP
construct the input of simi_rep
"""
repesentation = []
target = [] #age
c_x = []
c_y = []
# load model

General_Neuralprocess = NeuralProcess(x_dim, y_dim, r_dim, z_dim, h_dim)
General_Neuralprocess.load_state_dict(
    torch.load(r'F:\ANP_pytorch\trained_models\age_estimation\firstWholeTrained.ckpt'))
General_Neuralprocess.training = False


for idx, singlePersonDataset in enumerate(singlePersonDatasets):
    Data_loader = FGNetDataLoader(singlePersonDataset, batch_size=batch_size, shuffle=True)

    for x, y in Data_loader:
        c_x.append(x)
        c_y.append(y)
        repesentation.append(General_Neuralprocess.xy_to_rep(x, y))
        target.append(y)



"""
Train Simi_representation
"""

simi = Simi_repesentation(r_size = repesentation[0].size(1))
simiOptimizer = torch.optim.Adam(simi.parameters(), lr=1e-5)
simiEpoch = 300
criterion = torch.nn.MSELoss()
simi_loss_history = []
cnt = 0
for epoch in range(simiEpoch):
    context = []
    target_x = []
    target_y = []
    if cnt == num_of_people:
        cnt = 0
    for i in range(num_of_people):
        if i != cnt:
            context.append(repesentation[i])
    target_x = repesentation[cnt]
    target_y = target[cnt]
    x = c_x[cnt]
    y = c_y[cnt]


    simiOptimizer.zero_grad()
    context = torch.cat(context,dim=0)

    for r in context:
        simi_rep_list.append(simi(r, isContext=True))
    simi_rep_list = simi(context, isContext = False)
    print(simi_rep_list.size())
    context_simi_rep = torch.stack(simi_rep_list,dim=0)
    target_simi_rep = simi(target_x, isContext=False)
    mergeNet = MergeNet(context_simi_rep=context_simi_rep, target_simi_rep=target_simi_rep)
    weight = mergeNet.merge()

    num_context = 17
    num_target = 1
    resultsOnPretrainedModelsList = []
    for idx in range(num_of_people):
        if idx != cnt:
            # load model
            testNeuralprocess = NeuralProcess(x_dim, y_dim, r_dim, z_dim, h_dim)
            testModelPath = r'F:\ANP_pytorch\trained_models\age_estimation\smallTrained\smallTrained' + str(idx) + r'.ckpt'
            testNeuralprocess.load_state_dict(torch.load(testModelPath))
            testNeuralprocess.training = False

            x_context, y_context, _, _ = NeuralProcessTrainer.context_target_split(x, y, num_context, num_target)
            avg_mu = 0
            for i in range(10):
                p_y_pred = testNeuralprocess(x_context,y_context,x)
                # Extract mean of distribution
                mu = p_y_pred.loc.detach()
                avg_mu += mu
            avg_mu = avg_mu / 10
            avg_mu = avg_mu.view(18)
            resultsOnPretrainedModelsList.append(avg_mu.tolist())
    cnt += 1
    resultsOnPretrainedModelsList = torch.FloatTensor(resultsOnPretrainedModelsList)
    pred_age = torch.zeros(18,1)
    # column
    for j in range(len(resultsOnPretrainedModelsList[0])):
        # row
        for i in range(len(resultsOnPretrainedModelsList)):
            pred_age[j] += resultsOnPretrainedModelsList[i][j] * weight[j][i]
    target_y = target_y.view(num_of_images,1)
    pred_age = torch.FloatTensor(pred_age)
    pred_age = pred_age.view(num_of_images,1)
    loss = criterion(pred_age, target_y)
    loss.backward(retain_graph=True)
    simiOptimizer.step()
    print("Epoch: {}, loss: {}".format(epoch, loss))
    simi_loss_history.append(loss)
plt.plot(range(len(simi_loss_history)), simi_loss_history)
plt.show()
path = r'F:\ANP_pytorch\trained_models\Simi_rep.ckpt'
torch.save(simi.state_dict(), path)

