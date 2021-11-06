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




def calculate(model, loader):
    x = torch.linspace(-1,1,51)
    y = torch.linspace(-1,1,51)
    x,y = torch.meshgrid(x,y)
    x = x.reshape(-1)
    y = y.reshape(-1)
    xs = []
    ys = []
    uwu = []
    count = 0;
    tic = time.time()
    for xx,yy in zip(x,y):
        print(xx,yy)
        count += 1
        loss_md = LossSurfaceModel(model,xx,yy,device=device).to(device)
        xs.append(xx.item())
        ys.append(yy.item())
        uwu.append(loss_md.evaluate(loader)[0])
        print("Number {:05d}\tTime: {:0.5f} second(s)".format(count, time.time() - tic))
        tic = time.time()
    return xs,ys,uwu
MNIST = torchvision.datasets.MNIST("data/mnist")
_ds_len, _ds = preprocess_data(MNIST, sigma=50.0, device=device)
_ds, _ = torch.utils.data.random_split(_ds, [35000,25000])
loader = DataLoader(_ds, batch_size=12500)
xs,ys,uwu = calculate(Network,loader)
with open('result/loss_surface.json', 'w') as f:
    json.dump({"x": xs, "y":ys, "loss": uwu}, f)
