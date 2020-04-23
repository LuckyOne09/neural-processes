import numpy as np
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from datasets import FaceFeatureData

# Create dataset

dataset = FaceFeatureData(num_of_people=3,num_of_images=18)

#82 different people(batch_num)
#18 different images each people(batch_size)
#x_dim = 2048

from neural_process import NeuralProcess

x_dim = 2048
y_dim = 1
r_dim = 50  # Dimension of representation of context points
z_dim = 50  # Dimension of sampled latent variable
h_dim = 50  # Dimension of hidden layers in encoder and decoder

neuralprocess = NeuralProcess(x_dim, y_dim, r_dim, z_dim, h_dim)

from torch.utils.data import DataLoader
from training import NeuralProcessTrainer

batch_size = 1
num_context = 17
num_target = 1

data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
optimizer = torch.optim.Adam(neuralprocess.parameters(), lr=3e-4)
np_trainer = NeuralProcessTrainer(device, neuralprocess, optimizer,
                                  num_context_range=(num_context, num_context),
                                  num_extra_target_range=(num_target, num_target),
                                  print_freq=200)

neuralprocess.training = True
np_trainer.train(data_loader, 30)
