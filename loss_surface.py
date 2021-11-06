import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
import torchvision
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from torchdiffeq import odeint_adjoint as odeint
#from jupyterthemes import jtplot
from utils import *
import json
import pandas as pd
from pandas import  json_normalize 
import time
#jtplot.style(theme="chesterish")
 # CONSTANT 
device = "cuda"
EPOCHS=1
BATCH_SIZE=32
IMG_SIZE=(28,28)

cnn = Network().to(device)
ode = ODENet(ODEBlock().to(device),device=device).to(device)


MNIST = torchvision.datasets.MNIST("data/mnist")
_ds_len, _ds = preprocess_data(MNIST, sigma=50.0, device=device)
_ds, _ = torch.utils.data.random_split(_ds, [32768,60000-32768])
loader = DataLoader(_ds, batch_size=16384)
#xs,ys,uwu = calculate(Network,loader)


cnn_r = LossSurface.run(cnn,loader)
ode_r = LossSurface.run(ode,loader)
data = {}
data.update({
    "x": [x[0].item() for x in cnn_r],
    "y": [y[1].item() for y in cnn_r],
    "ode": {
        "loss": [l[2] for l in ode_r]
    },
    "cnn": {
        "loss": [l[2] for l in cnn_r]
    }
})
with open('result/loss_surface.json', 'w') as f:
    json.dump(data, f)

