import torch
from math import pi
import os

from datasets import FaceFeatureData

# Create dataset

ageVector = [[40, 22, 8, 18, 19, 43, 14, 33, 43, 16, 2, 28, 14, 29, 18, 5, 10, 2]]

ageTensor = torch.FloatTensor(ageVector)
print(ageTensor)