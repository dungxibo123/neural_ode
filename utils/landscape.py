import torch
import torch.nn as nn
import os
import sys
from torch._C import device
from model import *



device = "cuda"
sys.path.insert(0,os.path.abspath(__file__))

class LossSurface:
    @staticmethod
    def get_weight_as_list(net):
        """
        Get weight, not get "grad"
        """
        return [p.data for p in net.parameters()]

    @staticmethod
    def get_random_weight(weights):
        """
            give a direction for weights
        """
        return [torch.randn(p.size()).to(device) for p in weights]


    @staticmethod
    def get_oxy_surface_weight(net):
        NUMS = 51
        weights = LossSurface.get_weight_as_list(net)
        direction = [LossSurface.get_random_weight(weights) for i in range(2)]
        dx = direction[0]
        dy = direction[1]
        x,y = torch.linspace(-1,1, NUMS), torch.linspace(-1,1,NUMS)
        xs, ys = torch.meshgrid(x,y)
        param_groups = [] 
        for x_,y_ in zip(xs.reshape(-1),ys.reshape(-1)):
            step_ = [weights[i] + torch.mul(dx[i], x_.item()) + torch.mul(dy[i], y_.item()) for i in range(len(weights))]
            param_groups.append([x_,y_,step_]) 
             
        return param_groups
    @staticmethod
    def update_param(net,weight):
        for (p,w) in zip(net.parameters(), weight):
            p.data = w
    #t = ODENet(ODEBlock())
    #u = get_oxy_surface_weight(t)[1][2]
    #update_param(t,u)
    @staticmethod
    def run(net, loader):
        weights = LossSurface.get_oxy_surface_weight(net)
        uwu = []
        for weight in weights:
            LossSurface.update_param(net, weight[2])
            loss_,_ = net.evaluate(loader)
            uwu.append([weight[0],weight[1],loss_])
        return uwu
            

            

