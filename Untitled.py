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
#import matplotlib.pyplot as plt
from tqdm import tqdm
from torchdiffeq import odeint_adjoint as odeint
# from jupyterthemes import jtplot
# from neural_ode.utils import *
# jtplot.style(theme="chesterish")
 # CONSTANT 
device = "cuda"
EPOCHS=1
BATCH_SIZE=32
IMG_SIZE=(32,32,3)


# In[2]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint

import sys
import os

# sys.path.insert(0,os.path.abspath(__file__))

class ODEBlock(nn.Module):
    def __init__(self, parallel=None):
        super(ODEBlock,self).__init__()
        self.parallel = parallel
        self.conv1 = nn.Conv2d(64+1,64,3,1, padding=1)
        self.norm1 = nn.GroupNorm(32,64)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(64+1,64,3,1, padding=1)
        self.norm2 = nn.GroupNorm(32,64)
        
    def forward(self,t,x): 
        tt = torch.ones_like(x[:, :1, :, :]) * t
        out = torch.cat([tt, x], 1)
        out = self.conv1(out)
        out = self.norm1(out)
        out = self.relu(out)
        out = torch.cat([tt, out], 1)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)
        
        return out
     
class ODENet(nn.Module):
    def __init__(self, func, parallel=False, device="cpu"):
        super(ODENet, self).__init__()
        assert isinstance(func, ODEBlock) or isinstance(func.module,ODEBlock), f"argument function is not NeuralODEs model"
        self.fe = nn.Sequential(*[nn.Conv2d(3,16,3,1),
                                  nn.GroupNorm(16,16),
                                  nn.ReLU(),
                                  nn.Conv2d(16,32,3,2),
                                  nn.GroupNorm(32,32),
                                  nn.ReLU(),
                                  nn.Conv2d(32,64,4,2),
                                  nn.GroupNorm(32,64),
                                  #1x64x12x12
                                  nn.ReLU()])
        self.rm = func
        self.fcc = nn.Sequential(*[nn.AdaptiveAvgPool2d(1),
                                   # 1 x 64 x 1 x 1
                                   nn.Flatten(),
                                   nn.Linear(64,10),
                                   nn.Softmax()])
        self.intergrated_time = torch.Tensor([0.,1.]).float().to(device)
        self.parallel = parallel
    def forward(self,x):
        out = self.fe(x)
        self.intergrated_time = self.intergrated_time.to(out.device)
        if self.parallel:
            out = odeint(self.rm.module, out, self.intergrated_time, method="euler",options=dict(step_size=0.1), rtol=1e-3, atol=1e-3)[1]
        else:
            out = odeint(self.rm, out, self.intergrated_time, method="euler",options=dict(step_size=0.1), rtol=1e-3, atol=1e-3)[1]
        
        #out = self.rm(out)
        out = self.fcc(out)
        return out
    def evaluate(self, test_loader):
        correct = 0
        total = 0 
        running_loss = 0
        count = 0        
        with torch.no_grad():
            for batch_id , test_data in enumerate(test_loader,0):
                count += 1
                data, label = test_data
                outputs = self.forward(data)
                _, correct_labels = torch.max(label, 1) 
                _, predicted = torch.max(outputs.data, 1)
                total += label.size(0)
                correct += (predicted == correct_labels).sum().item()
                running_loss += F.torch.nn.functional.binary_cross_entropy_with_logits(
                    outputs.float(), label.float()).item()
        #        print(f"--> Total {total}\n-->batch_id: {batch_id + 1}")
        acc = correct/total
        running_loss /= count 
        return running_loss,acc

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.fe = nn.Sequential(*[
            nn.Conv2d(3,16,3,1),
            nn.GroupNorm(16,16),
            nn.ReLU(),
            nn.Conv2d(16,32,3,2),
            nn.GroupNorm(32,32),
            nn.ReLU(),
            nn.Conv2d(32,64,3,2),
            nn.GroupNorm(32,64),
            nn.ReLU()
             
        ])
        self.rm = nn.Sequential(*[
            nn.Conv2d(64,64,3,1, padding=1),
            nn.GroupNorm(32,64),
            nn.ReLU(), 
            nn.Conv2d(64,64,3,1, padding=1),
            nn.GroupNorm(32,64),
            nn.ReLU(),
        ])
        self.fcc = nn.Sequential(*[
            #nn.Conv2d(64,1,3,1,padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64,10),
            nn.Softmax()
        ])
    def forward(self,x):
        out = self.fe(x)
        out = out + self.rm(out)
        out = self.fcc(out)
        return out

        return self.net(x)
    def evaluate(self, test_loader):
        correct = 0
        total = 0 
        running_loss = 0
        count = 0 
        with torch.no_grad():
            for test_data in test_loader:
                count += 1
                data, label = test_data
                outputs = self.forward(data)
                _, correct_labels = torch.max(label, 1) 
                _, predicted = torch.max(outputs.data, 1)
                total += label.size(0)
                correct += (predicted == correct_labels).sum().item()
                running_loss += F.torch.nn.functional.binary_cross_entropy_with_logits(
                    outputs.float(), label.float()).item()
        acc = correct / total
        running_loss /= count
        
        return running_loss,acc


# In[3]:


def train_model(model, optimizer, train_loader, val_loader,loss_fn, lr_scheduler=None, epochs=100, parallel=None):
    #print(model.eval())
    model.train()
    print(f"Numbers of parameters in model: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    best_model, best_acc, best_epoch = None, 0, 0
    history = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}
    for epoch_id in tqdm(range(epochs)):
        total = 0
        correct = 0
        running_loss = 0
        print(f"Start epoch number: {epoch_id + 1}")
#        print(next(enumerate(train_loader,0)))
        loads = list(enumerate(train_loader,0))
        for batch_id, data in loads:
#            print("Go here please")
            # get the inputs; data is a list of [inputs, labels]
            #print(f"Start batch number: {batch_id + 1} in epoch number: {epoch_id + 1}")
            inputs, labels = data
#            print(f"This is labels: {labels}\n\n\n\n")
            #print(f"Get data done")
            # zero the parameter gradients
            optimizer.zero_grad()
            #print(f"Reset the optimizer backward, grad to 0") 
            # forward + backward + optimize
            outputs = model(inputs)
            #print(f"forward data through model")
            _, predicted = torch.max(outputs, 1)
            #print(f"Get predicted class")
            _, correct_labels = torch.max(labels, 1)
            #print(f"Get label class")
            #print(labels)
            total += labels.size(0)
            correct += (predicted == correct_labels).sum().item()
            #print("Calculate the number of correct predictions")
            #print(labels.shape, outputs.shape)
            loss = loss_fn(outputs.float(), labels.float())
            loss.backward()
            #print("Backward loss")
            optimizer.step()
            #print("Step")
            running_loss += loss.item() 
            #print("End batch number: {batch_id + 1} in epoch number {epoch_id + 1}")
        #acc = round(correct/total * 1.0, 5)
        acc = correct / total

        #print("Accuracy was calculated")
        history["acc"].append(acc)
        history["loss"].append(running_loss)
        if parallel is not None:
            val_loss, val_acc = model.module.evaluate(val_loader)
        else:
            val_loss, val_acc = model.evaluate(val_loader)
        if val_acc > best_acc:
            best_acc = acc
            best_epoch = epoch_id + 1
            best_model = model
        if lr_scheduler is not None:
            lr_scheduler.step(val_loss)        
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        running_loss /= len(loads)
        #print(f"Epoch(s) {epoch_id + 1} | loss: {loss} | acc: {acc} | val_loss: {val_loss} | val_acc: {val_acc}")
        # checkpoint = {
        #     'epoch': epoch_id + 1,
        #     'model': model,
        #     'best_epoch': best_epoch,
        #     'optimizer': optimizer.state_dict()
        # }
        # torch.save(checkpoint, "./checkpoints/checkpoint.pt")
        print("Epoch(s) {:04d}/{:04d} | acc: {:.05f} | loss: {:.09f} | val_acc: {:.05f} | val_loss: {:.09f} | Best epochs: {:04d} | Best acc: {:09f}".format(
            epoch_id + 1, epochs, acc, running_loss, val_acc, val_loss, best_epoch, best_acc
            ))

    return history, best_model, best_epoch, best_acc



def main(ds_len, train_ds, valid_ds,model_type = "ode",data_name = "mnist_50",batch_size=32,epochs=100, lr=1e-3,train_num = 0, valid_num = 0, test_num = 0, weight_decay=0.1, device="cpu", result_dir="./result", model_dir="./model", parallel=None):
    print(f"Number of train: {train_num}\nNumber of validation: {valid_num}")
    #train_set = torch.utils.data.random_split(ds)
    #print(type(train_set))
    #assert isinstance(train_set,torch.utils.data.Dataset)
    #train_ds, _ = torch.utils.data.random_split(train_ds, lengths=[TRAIN_NUM, ds_len - TRAIN_NUM])
    #valid_ds, _ = torch.utils.data.random_split(valid_ds, lengths=[VALID_NUM, ds_len - VALID_NUM])
    print(len(train_ds))
    train_loader = DataLoader(train_ds, shuffle=True, batch_size=batch_size, drop_last=True)
    val_loader  = DataLoader(valid_ds, shuffle=True, batch_size= batch_size * 16, drop_last=True)
    loss_fn = torch.nn.functional.binary_cross_entropy_with_logits
    if parallel is not None:
        if model_type == "ode": 
            ode_func = ODEBlock(parallel=parallel)
            ode_func = nn.DataParallel(ode_func).to(device)
            model = ODENet(ode_func, parallel, device=device)
            model = nn.DataParallel(model).to(device)
#    ode_func = DDP(ODEBlock().to(device), output_device=device)
#    ode_model = DDP(ODENet(ode_func,device=device).to(device),output_device=device)
        elif model_type == "cnn":
#            epochs= int(epochs * 1.5)
            model = Network()
            model = nn.DataParallel(model).to(device)
    else:
        if model_type == "ode": 
            ode_func = ODEBlock().to(device)
            ode_func = ode_func.to(device)
            model = ODENet(ode_func, device=device)
            model = model.to(device)
#    ode_func = DDP(ODEBlock().to(device), output_device=device)
#    ode_model = DDP(ODENet(ode_func,device=device).to(device),output_device=device)
        elif model_type == "cnn":
            #epochs= int(epochs * 1.5)
            model = Network().to(device)
            #model = nn.DataParallel(model).to(device)
        

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = None
    if weight_decay is not None:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=weight_decay, patience=5)
    his, model, epoch, acc = train_model(model, 
                      optimizer, 
                      train_loader,
                      val_loader,
                      lr_scheduler=lr_scheduler,
                      loss_fn=loss_fn, 
                      epochs=epochs,
                      parallel=parallel)
     
    # save_result(his,model_name=model_type,ds_name=data_name, result_dir=result_dir)
    # if not os.path.exists(f"{MODEL_DIR}/{model_type}_origin"):
    #     os.mkdir(f"{MODEL_DIR}/{model_type}_origin")
    # print("Save original data modeling...")
    # torch.save(model.state_dict(), f"{MODEL_DIR}/{model_type}_origin/{data_name}_origin.pt" ) 
    return model


# In[4]:


def add_noise(converted_data, sigma = 10,device="cpu"):
    pertubed_data = converted_data + torch.normal(torch.zeros(converted_data.shape),
                                                  torch.ones(converted_data.shape) * sigma).to(device)
    #pertubed_data = torch.tensor(random_noise(converted_data.cpu(), mode='gaussian', mean=0, var=sigma**2, clip=False)).float().to(device)
    return pertubed_data
def preprocess_data(data, shape = (28,28), sigma=None,device="cpu", train=False):
    if not train:
        #assert type(sigma) == type(list()) or type(sigma) == type(None), f"if train=False, the type(sigma) must be return a list object or NoneType object, but return {type(sigma)}"
        X = []
        Y = []
        ds = {}
        sigma_noise = [50.,75.,100.]
        for data_idx, (x,y) in list(enumerate(data)):
            #X.append(np.array(x).reshape((3,shape[0],shape[0])))
            X.append(np.array(x).transpose(2,0,1)) # Change the shape from (H,W,C) -> (C,H,W)
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
    else:
        import random
        X = []
        Y = []
        for data_idx, (x, y) in list(enumerate(data)):
            std = random.choice(sigma)
            noise_x = (np.array(x) + np.random.normal(np.zeros_like(np.array(x)), np.ones_like(np.array(x)) * std))
            X.append(noise_x.transpose(2,0,1))
            Y.append(y)
        y_data = F.one_hot(torch.Tensor(Y).to(torch.int64), num_classes=10)
        y_data = y_data.to(device)
        x_data = torch.Tensor(X)
        x_data = x_data.to(device)
        return len(Y), TensorDataset(x_data,y_data) 


# In[5]:


BATCH_SIZE = 128
EPOCHS = 120
TRAIN_NUM = VALID_NUM = TEST_NUM = 0
MNIST = torchvision.datasets.CIFAR10('./data',
                                   train=True,
                                   transform=None,
                                   target_transform=None, download=True)

ds_len_, ds_ = preprocess_data(MNIST, sigma=None, device=device)
ds_len_, pertubed_ds_ = preprocess_data(MNIST, sigma=[15.0], device=device, train=True)
print(type(ds_))
    
sigma = [None, 1e-7, 15.0, 20.0, 50.0, 75.0, 100.0]
loaders = [(key,DataLoader(preprocess_data(MNIST, sigma=key, device=device, train=False)[1], batch_size=12000)) for key in sigma]


# In[6]:


evaluation = {
    "ode": {
        
    },
    "cnn": {

    }
}
for k in sigma:
    evaluation["ode"].update({k: []})
    evaluation["cnn"].update({k: []})


# In[7]:


print(device)
EPOCHS = 120


# In[ ]:


for i in range(1):
    cnn_model = main(ds_len_,ds_, pertubed_ds_, device=device, model_type="cnn", data_name=f"mnist_origin",batch_size=BATCH_SIZE, epochs=EPOCHS, train_num=TRAIN_NUM, valid_num=VALID_NUM, test_num=TEST_NUM, parallel=None) 
    ode_model = main(ds_len_,ds_, pertubed_ds_, device=device, model_type="ode", data_name=f"mnist_origin",batch_size=BATCH_SIZE, epochs=EPOCHS, train_num=TRAIN_NUM, valid_num=VALID_NUM, test_num=TEST_NUM, parallel=None) 
    for k,l in loaders:
        if isinstance(cnn_model, nn.DataParallel): cnn_model = cnn_model.module
        if isinstance(ode_model, nn.DataParallel): ode_model = ode_model.module
        _, cnn_acc = cnn_model.evaluate(l) 
        _, ode_acc = ode_model.evaluate(l) 
        
        print(f"CNNs for {k}-gaussian-pertubed CIFAR10 = {cnn_acc}")
        print(f"ODEs for {k}-gaussian-pertubed CIFAR10 = {ode_acc}")
        

        evaluation["ode"][k].append(ode_acc)
        evaluation["cnn"][k].append(cnn_acc)

import json

with open('results_pertubed.json', 'w') as fp:
    json.dump(evaluation, fp)

        
# In[8]:
evaluation = {
    "ode": {
        
    },
    "cnn": {

    }
}
for k in sigma:
    evaluation["ode"].update({k: []})
    evaluation["cnn"].update({k: []})


for i in range(1):
    cnn_model = main(ds_len_,ds_, ds_, device=device, model_type="cnn", data_name=f"mnist_origin",batch_size=BATCH_SIZE, epochs=EPOCHS, train_num=TRAIN_NUM, valid_num=VALID_NUM, test_num=TEST_NUM, parallel=None) 
    ode_model = main(ds_len_,ds_, ds_, device=device, model_type="ode", data_name=f"mnist_origin",batch_size=BATCH_SIZE, epochs=EPOCHS, train_num=TRAIN_NUM, valid_num=VALID_NUM, test_num=TEST_NUM, parallel=None) 
    for k,l in loaders:
        if isinstance(cnn_model, nn.DataParallel): cnn_model = cnn_model.module
        if isinstance(ode_model, nn.DataParallel): ode_model = ode_model.module
        _, cnn_acc = cnn_model.evaluate(l) 
        _, ode_acc = ode_model.evaluate(l) 
        
        print(f"CNNs for {k}-gaussian-pertubed CIFAR10 = {cnn_acc}")
        print(f"ODEs for {k}-gaussian-pertubed CIFAR10 = {ode_acc}")
        

        evaluation["ode"][k].append(ode_acc)
        evaluation["cnn"][k].append(cnn_acc)

with open('results.json', 'w') as fp:
    json.dump(evaluation, fp)

# In[ ]:




