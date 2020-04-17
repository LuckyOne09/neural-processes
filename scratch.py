import numpy as np
import torch

# locations = np.random.choice(100,
#                              size=4 + 2,
#                              replace=False)
# print(locations)
#
# a = np.arange(100)
# print(a)

b = torch.tensor([1,2,3])
print(b)

print(b.contiguous())