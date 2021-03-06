import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.util import random_noise
from torch.utils.data import TensorDataset
import torchvision
import os.path
import json
sys.path.insert(0,os.path.abspath(__file__))

def save_result(his, model_name = "ode",ds_name="mnist_0", result_dir="./result"):
    if not os.path.exists(f"./{result_dir}"):
        os.mkdir(f"./{result_dir}")
    if not os.path.exists(f"{result_dir}/{model_name}_results"):
        os.mkdir(f"{result_dir}/{model_name}_results") 
    with open(f'{result_dir}/{model_name}_results/{ds_name}.json', 'w', encoding='utf-8') as fo:
        json.dump(his, fo, indent=4)

def add_noise(converted_data, sigma = 10,device="cpu"):
    pertubed_data = converted_data + torch.normal(torch.zeros(converted_data.shape),
                                                  torch.ones(converted_data.shape) * sigma).to(device)
    #pertubed_data = torch.tensor(random_noise(converted_data.cpu(), mode='gaussian', mean=0, var=sigma**2, clip=False)).float().to(device)
    return pertubed_data
def preprocess_data(data, shape = (28,28), sigma=None,device="cpu"):
    X = []
    Y = []
    ds = {}
    sigma_noise = [50.,75.,100.]
    for data_idx, (x,y) in list(enumerate(data)):
        X.append(np.array(x).reshape((1,shape[0],shape[0])))
        Y.append(y)
    y_data = F.one_hot(torch.Tensor(Y).to(torch.int64), num_classes=10)
    y_data = y_data.to(device)
    x_data = torch.Tensor(X)
    x_data = x_data.to(device)
    if sigma:
        x_noise_data = add_noise(x_data, sigma=sigma, device=device) / 255.0
        print(f"Generating {sigma}-pertubed-dataset")
    else:
        x_noise_data = x_data
        print(f"Generating {sigma}-pertubed-dataset")

    pertubed_ds = TensorDataset(x_noise_data,y_data)
    #ds.update({"original": TensorDataset(x_data / 255.0, y_data)})
    ds_len = len(Y)
    return ds_len, pertubed_ds
