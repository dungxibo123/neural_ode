#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Necessary
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
from jupyterthemes import jtplot
from utils import *
jtplot.style(theme="chesterish")
 # CONSTANT 
device = "cpu"
EPOCHS=1
BATCH_SIZE=32
IMG_SIZE=(32,32,3)


# In[2]:


# Load data
DIR = "./data/mnist/"
MNIST = torchvision.datasets.MNIST(DIR,
                                   train=True,
                                   transform=None,
                                   target_transform=None, download=False)


#ds_len_, normal_ds_, pertubed_ds_ = preprocess_data(MNIST)


# In[3]:


cnn_model = Network()
ode_func = ODEBlock()
ode_model = ODENet(ode_func)


# In[4]:


def model_state_dict_parallel_convert(state_dict, mode):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    if mode == 'to_single':
        for k, v in state_dict.items():
            name = k[7:]  # remove 'module.' of DataParallel
            new_state_dict[name] = v
    elif mode == 'to_parallel':
        for k, v in state_dict.items():
            name = 'module.' + k  # add 'module.' of DataParallel
            new_state_dict[name] = v
    elif mode == 'same':
        new_state_dict = state_dict
    else:
        raise Exception('mode = to_single / to_parallel')

    return new_state_dict 
cnn_state_dict = torch.load("./model/cnn_origin/mnist_original_origin.pt",map_location=torch.device('cpu'))
cnn_state_dict = model_state_dict_parallel_convert(cnn_state_dict, mode="to_single")
cnn_model.load_state_dict(cnn_state_dict)
ode_state_dict = torch.load("./model/ode_origin/mnist_original_origin.pt",map_location=torch.device('cpu'))
ode_state_dict = model_state_dict_parallel_convert(ode_state_dict, mode="to_single")
ode_model.load_state_dict(ode_state_dict)


# In[5]:


_ds_len, _ds = preprocess_data(MNIST)

print(_ds)


# In[7]:


for key in _ds.keys():
    loader = DataLoader(_ds[key], batch_size=10000)
    _, cnn_acc = cnn_model.evaluate(loader)
    _, ode_acc = ode_model.evaluate(loader)
    print(f"CNNs for {key}-gaussian-pertubed MNIST = {cnn_acc}")
    print(f"ODEs for {key}-gaussian-pertubed MNIST = {ode_acc}")


# In[ ]:




