import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
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
args = parser.parse_args()


 # CONSTANT 
device = args.device
#torch.device("cuda")
EPOCHS=args.epochs
BATCH_SIZE=args.batch_size
DATA_DIR = args.folder
RESULT_DIR = args.result
TRAIN_NUM=args.train
VALID_NUM=args.valid
TEST_NUM=60000-TRAIN_NUM-VALID_NUM
DATA_DISTRIBUTION=[TRAIN_NUM,VALID_NUM,TEST_NUM]




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
        acc = round(correct/total * 1.0, 2)
        #print("Accuracy was calculated")
        history["acc"].append(acc)
        history["loss"].append(running_loss)
        print("Before evaluate")
        val_loss, val_acc = model.evaluate(val_loader)
        print("After evaluation")
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"Epoch(s) {epoch_id + 1} | loss: {loss} | acc: {acc} | val_loss: {val_loss} | val_acc: {val_acc}")
    return history




def main(ds_len, ds, name = "mnist_50",batch_size=32,epochs=100, lr=1e-3,data_dis=[8000,2000,50000], device="cpu", result_dir="./result"):
    print(f"Number of train: {data_dis[0]}\nNumber of validation: {data_dis[1]}")
    train_set, val_set, _ = torch.utils.data.random_split(ds,data_dis)
    #print(type(train_set))
    assert isinstance(train_set,torch.utils.data.Dataset)
    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_set, shuffle=True, batch_size=data_dis[1])
    loss_fn = torch.nn.functional.binary_cross_entropy_with_logits
    ode_func = ODEBlock().to(device)
    ode_model = ODENet(ode_func, device=device).to(device)
    ode_optimizer = torch.optim.Adam(ode_model.parameters(), lr=lr)
    cnn_model = Network().to(device)
    cnn_optimizer = torch.optim.Adam(cnn_model.parameters(), lr=lr)
    cnn_his = train_model(cnn_model, 
                         cnn_optimizer,
                         train_loader, val_loader, loss_fn=loss_fn,epochs=epochs)
    ode_his = train_model(ode_model,
                          ode_optimizer,
                          train_loader, val_loader, loss_fn=loss_fn, epochs=epochs)
    save_result(cnn_his,model_name="cnn",ds_name=name, result_dir=result_dir)
    save_result(ode_his,model_name="ode",ds_name=name, result_dir=result_dir)


MNIST = torchvision.datasets.MNIST(DATA_DIR,
                                   train=True,
                                   transform=None,
                                   target_transform=None, download=True)

ds_len_, ds_ = preprocess_data(MNIST, device=device)

print(type(ds_))
for (sigma, ds) in ds_.items():
    main(ds_len_,ds, device=device, name=f"mnist_{sigma}",batch_size=BATCH_SIZE, epochs=EPOCHS, data_dis=DATA_DISTRIBUTION, result_dir=RESULT_DIR)

    
    




