import utils


from utils import *
import torch

model = ODENet(ODEBlock())

t = torch.rand((1,1,28,28))
print(model(t).shape)
