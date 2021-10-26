import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision
from tqdm import tqdm
import os.path
import argparse

import json
from utils import *



parser = argparse.ArgumentParser()
parser.add_argument("-d", "--device", type=str, default="cpu", help="Device which the PyTorch run on")
parser.add_argument("-bs", "--batch-size", type=int, default=1024, help="Batch size of 1 iteration")
parser.add_argument("-ep", "--epochs", type=int, default=200, help="Numbers of epoch")
parser.add_argument("-f", "--folder", type=str, default="./data/mnist", help="Folder /path/to/mnist/dataset")
parser.add_argument("-r", "--result", type=str, default="./result", help="Folder where the result going in")
parser.add_argument("-tr", "--train", type=int, default=8000, help="Number of train images")
parser.add_argument("-vl", "--valid", type=int, default=2000, help="Number of validation images")
parser.add_argument("-lr", "--learning-rate",type=float, default=1e-3, help="Learning rate in optimizer")
parser.add_argument("-md", "--model", type=str, default="./model", help="Where model going to")
args = parser.parse_args()


 # CONSTANT 
device = args.device
#torch.device("cuda")
EPOCHS=args.epochs
BATCH_SIZE=args.batch_size
DATA_DIR=args.folder
RESULT_DIR=args.result
TRAIN_NUM=args.train
VALID_NUM=args.valid
TEST_NUM=60000-TRAIN_NUM-VALID_NUM
DATA_DISTRIBUTION=[TRAIN_NUM,VALID_NUM,TEST_NUM]
MODEL_DIR=args.model




def train_model(model, optimizer, train_loader, val_loader,loss_fn, epochs=100):
    print(model.eval())
    print(f"Numbers of parameters in model: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    history = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}
    for epoch_id in tqdm(range(epochs)):
        total = 0
        correct = 0
        running_loss = 0
        print(f"Start epoch number: {epoch_id + 1}")
        for batch_id, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            #print(f"Start batch number: {batch_id + 1} in epoch number: {epoch_id + 1}")
            inputs, labels = data
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
            #print(f"End batch number: {batch_id + 1} in epoch number {epoch_id + 1}")
        acc = round(correct/total * 1.0, 5)
        #print("Accuracy was calculated")
        history["acc"].append(acc)
        history["loss"].append(running_loss)
        print("Before evaluate")
        val_loss, val_acc = model.module.evaluate(val_loader)
        print("After evaluation")
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"Epoch(s) {epoch_id + 1} | loss: {loss} | acc: {acc} | val_loss: {val_loss} | val_acc: {val_acc}")
    return history




def main(ds_len, ds,model_type = "ode",data_name = "mnist_50",batch_size=32,epochs=100, lr=1e-3,data_dis=[8000,2000,50000], device="cpu", result_dir="./result", model_dir="./model"):
    print(f"Number of train: {data_dis[0]}\nNumber of validation: {data_dis[1]}")
    train_set, val_set, _ = torch.utils.data.random_split(ds,data_dis)
    #print(type(train_set))
    assert isinstance(train_set,torch.utils.data.Dataset)
    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_set, shuffle=True, batch_size=data_dis[1])
    loss_fn = torch.nn.functional.binary_cross_entropy_with_logits
    if model_type == "ode": 
        ode_func = ODEBlock()
        ode_func = nn.DataParallel(ode_func).to(device)
        model = ODENet(ode_func.module, device=device)
        model = nn.DataParallel(model).to(device)
#    ode_func = DDP(ODEBlock().to(device), output_device=device)
#    ode_model = DDP(ODENet(ode_func,device=device).to(device),output_device=device)
    elif model_type == "cnn":
        epochs= int(epochs * 1.5)
        model = Network()
        model = nn.DataParallel(model).to(device)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    his = train_model(model, 
                      optimizer, 
                      train_loader,
                      val_loader,
                      loss_fn=loss_fn, 
                      epochs=epochs)

    save_result(his,model_name=model_type,ds_name=data_name, result_dir=result_dir)
    if data_name.split("_")[-1] == "original":
        if not os.path.exists(f"{MODEL_DIR}/{model_type}_origin"):
            os.mkdir(f"{MODEL_DIR}/{model_type}_origin")
        print("Save original data modeling...")
        torch.save(model.state_dict(), f"{MODEL_DIR}/{model_type}_origin/{data_name}_origin.pt" ) 

MNIST = torchvision.datasets.MNIST(DATA_DIR,
                                   train=True,
                                   transform=None,
                                   target_transform=None, download=True)

ds_len_, ds_ = preprocess_data(MNIST, device=device)

print(type(ds_))
for (sigma, ds) in ds_.items(): 
    if sigma.split("_")[-1] == "original":
        main(ds_len_,ds, device=device, model_type="cnn", data_name=f"mnist_{sigma}",batch_size=BATCH_SIZE, epochs=EPOCHS, data_dis=DATA_DISTRIBUTION, result_dir=RESULT_DIR) 
for (sigma,ds) in ds_.items():
    if sigma.split("_")[-1] == "original":
        main(ds_len_,ds, device=device, model_type="ode", data_name=f"mnist_{sigma}",batch_size=BATCH_SIZE, epochs=EPOCHS, data_dis=DATA_DISTRIBUTION, result_dir=RESULT_DIR)

    
    




