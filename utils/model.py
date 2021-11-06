import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint

import sys
import os

sys.path.insert(0,os.path.abspath(__file__))

class ODEBlock(nn.Module):
    def __init__(self, parallel=None):
        super(ODEBlock,self).__init__()
        self.parallel = parallel
        self.block = nn.Sequential(*[
            nn.Conv2d(64,64,3,1, padding=1),
            nn.GroupNorm(32,64),
            nn.ReLU(), 
            nn.Conv2d(64,64,3,1, padding=1),
            nn.GroupNorm(16,64),
            nn.ReLU()
        ]) 
    def forward(self,t,x): 
        return self.block(x)
     
class ODENet(nn.Module):
    def __init__(self, func, parallel=False, device="cpu"):
        super(ODENet, self).__init__()
        assert isinstance(func, ODEBlock) or isinstance(func.module,ODEBlock), f"argument function is not NeuralODEs model"
        self.fe = nn.Sequential(*[nn.Conv2d(1,64,3,1),
                                  nn.GroupNorm(32,64),
                                  nn.ReLU(),
                                  nn.Conv2d(64,64,4,2),
                                  nn.GroupNorm(16,64),
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
            out = odeint(self.rm.module, out, self.intergrated_time, method="rk4",options=dict(step_size=0.1))[1]
        else:
            out = odeint(self.rm, out, self.intergrated_time, method="rk4",options=dict(step_size=0.1))[1]
        
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
        acc = round(correct/total * 1.0 , 5)
        running_loss /= count 
        return running_loss,acc

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.fe = nn.Sequential(*[
            nn.Conv2d(1,64,3,1),
            nn.GroupNorm(32,64),
            nn.ReLU(),
            nn.Conv2d(64,64,4,2),
            nn.GroupNorm(16,64),
            nn.ReLU()
             
        ])
        self.rm = nn.Sequential(*[
            nn.Conv2d(64,64,3,1, padding=1),
            nn.GroupNorm(32,64),
            nn.ReLU(), 
            nn.Conv2d(64,64,3,1, padding=1),
            nn.GroupNorm(16,64),
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
        acc = round(correct/total * 1.0 , 5)
        running_loss /= count
        
        return running_loss,acc


class LossSurfaceModel(nn.Module):
    def __init__(self, model, a, b, device="cpu"):
        super(LossSurfaceModel, self).__init__()
        self.a = a
        self.b = b
        if model == ODENet:
            print("Loss on ODENet")
            self.net1 = model(ODEBlock().to(device)).to(device)
            self.net2 = model(ODEBlock().to(device)).to(device)
        else:
            print("Loss on CNNNet")
            self.net1 = model().to(device)
            self.net2 = model().to(device)
    def forward(self,x):
        return self.a.item() * self.net1(x) + self.b.item() * self.net2(x)
    def evaluate(self,test_loader):
        #def evaluate(self, test_loader):
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
