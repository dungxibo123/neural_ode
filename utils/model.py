import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint

import sys
import os

sys.path.insert(0,os.path.abspath(__file__))

class ODEBlock(nn.Module):
    def __init__(self):
        super(ODEBlock,self).__init__()
        self.block = nn.Sequential(*[
            nn.Conv2d(64,64,3,1, padding=1),
            nn.GroupNorm(4,64),
            nn.ReLU(), 
            nn.Conv2d(64,64,3,1, padding=1),
            nn.GroupNorm(4,64),
            nn.ReLU()
        ]) 
    def forward(self,t,x):
        return self.block(x)
     
class ODENet(nn.Module):
    def __init__(self, func, device="cpu"):
        super(ODENet, self).__init__()
        assert isinstance(func, ODEBlock), f"argument function is not NeuralODEs model"
        self.fe = nn.Sequential(*[nn.Conv2d(1,64,3,1),
                                  nn.GroupNorm(4,64),
                                  nn.ReLU(),
                                  nn.Conv2d(64,64,4,2),
                                  nn.GroupNorm(4,64),
                                  #1x64x12x12
                                  nn.ReLU()])
        self.rm = func
        self.fcc = nn.Sequential(*[nn.Conv2d(64,1,3,1,padding=1),
                                   nn.AdaptiveAvgPool2d(8),
                                   # 1 x 64 x 8 x 8
                                   nn.Flatten(),
                                   # 4096
                                   nn.Linear(64,10),
                                   nn.Softmax()])
        self.intergrated_time = torch.Tensor([0.,1.]).float().to(device)
    def forward(self,x):
        out = self.fe(x)
        self.intergrated_time = self.intergrated_time.to(out.device)
        out = odeint(self.rm, out, self.intergrated_time)[1]
        
        #out = self.rm(out)
        out = self.fcc(out)
        return out
    def evaluate(self, test_loader):
        correct = 0
        total = 0 
        running_loss = 0
        
        with torch.no_grad():
            for batch_id , test_data in enumerate(test_loader,0):
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
         
        return running_loss,acc

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.fe = nn.Sequential(*[
            nn.Conv2d(1,64,3,1),
            nn.GroupNorm(4,64),
            nn.ReLU(),
            nn.Conv2d(64,64,4,2),
            nn.GroupNorm(4,64),
            nn.ReLU()
             
        ])
        self.rm = nn.Sequential(*[
            nn.Conv2d(64,64,3,1, padding=1),
            nn.GroupNorm(4,64),
            nn.ReLU(), 
            nn.Conv2d(64,64,3,1, padding=1),
            nn.GroupNorm(4,64),
            nn.ReLU(),
        ])
        self.fcc = nn.Sequential(*[
            nn.Conv2d(64,1,3,1,padding=1),
            nn.AdaptiveAvgPool2d(8),
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
        
        with torch.no_grad():
            for test_data in test_loader:
                data, label = test_data
                outputs = self.forward(data)
                _, correct_labels = torch.max(label, 1) 
                _, predicted = torch.max(outputs.data, 1)
                total += label.size(0)
                correct += (predicted == correct_labels).sum().item()
                running_loss += F.torch.nn.functional.binary_cross_entropy_with_logits(
                    outputs.float(), label.float()).item()
        acc = round(correct/total * 1.0 , 5)
        
        return running_loss,acc

