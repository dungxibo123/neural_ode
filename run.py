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
parser.add_argument("-pr", "--parallel", type=bool, default=False, help="Parallel or not")
parser.add_argument("-wd", "--weight-decay", type=float, default=0.1, help="Weight decay")
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
PARALLEL=args.parallel
WEIGHT_DECAY=args.weight_decay



def train_model(model, optimizer, train_loader, val_loader,loss_fn, lr_scheduler=None, epochs=100, parallel=None):
    #print(model.eval())
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
        if lr_scheduler:
            lr_scheduler.step()
        acc = correct / total

        #print("Accuracy was calculated")
        history["acc"].append(acc)
        history["loss"].append(running_loss)
        if parallel is not None:
            val_loss, val_acc = model.module.evaluate(val_loader)
        else:
            val_loss, val_acc = model.evaluate(val_loader)
        if acc > best_acc and val_acc > 0.85:
            best_acc = acc
            best_epoch = epoch_id + 1
            best_model = model
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        running_loss /= len(loads)
        #print(f"Epoch(s) {epoch_id + 1} | loss: {loss} | acc: {acc} | val_loss: {val_loss} | val_acc: {val_acc}")
        checkpoint = {
            'epoch': epoch_id + 1,
            'model': model,
            'best_epoch': best_epoch,
            'optimizer': optimizer.state_dict()
        }
        torch.save(checkpoint, "./checkpoints/checkpoint.pt")
        print("Epoch(s) {:04d}/{:04d} | acc: {:.05f} | loss: {:.09f} | val_acc: {:.05f} | val_loss: {:.09f} | Best epochs: {:04d} | Best acc: {:09f}".format(
            epoch_id + 1, epochs, acc, running_loss, val_acc, val_loss, best_epoch, best_acc
            ))

    return history, best_model, best_epoch, best_acc



def main(ds_len, train_ds, valid_ds,model_type = "ode",data_name = "mnist_50",batch_size=32,epochs=100, lr=1e-3,train_num = 0, valid_num = 0, test_num = 0, weight_decay=None, device="cpu", result_dir="./result", model_dir="./model", parallel=None):
    print(f"Number of train: {train_num}\nNumber of validation: {valid_num}")
    #train_set = torch.utils.data.random_split(ds)
    #print(type(train_set))
    #assert isinstance(train_set,torch.utils.data.Dataset)
    #train_ds, _ = torch.utils.data.random_split(train_ds, lengths=[TRAIN_NUM, ds_len - TRAIN_NUM])
    #valid_ds, _ = torch.utils.data.random_split(valid_ds, lengths=[VALID_NUM, ds_len - VALID_NUM])
    print(len(train_ds))
    train_loader = DataLoader(train_ds, shuffle=True, batch_size=batch_size, drop_last=True)
    val_loader  = DataLoader(valid_ds, shuffle=True, batch_size= batch_size * 16, drop_last=True)
    loss_fn = torch.nn.functional.cross_entropy
    if parallel:
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
            ode_func = nn.DataParallel(ode_func).to(device)
            model = ODENet(ode_func.module, device=device)
            model = nn.DataParallel(model).to(device)
#    ode_func = DDP(ODEBlock().to(device), output_device=device)
#    ode_model = DDP(ODENet(ode_func,device=device).to(device),output_device=device)
        elif model_type == "cnn":
            #epochs= int(epochs * 1.5)
            model = Network().to(device)
            #model = nn.DataParallel(model).to(device)
        

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if weight_decay:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=weight_decay, patience=5)
    his, model, epoch, acc = train_model(model, 
                      optimizer, 
                      train_loader,
                      val_loader,
                      lr_scheduler=lr_scheduler,
                      loss_fn=loss_fn, 
                      epochs=epochs,
                      parallel=parallel)
     
    save_result(his,model_name=model_type,ds_name=data_name, result_dir=result_dir)
    if not os.path.exists(f"{MODEL_DIR}/{model_type}_origin"):
        os.mkdir(f"{MODEL_DIR}/{model_type}_origin")
    print("Save original data modeling...")
    torch.save(model.state_dict(), f"{MODEL_DIR}/{model_type}_origin/{data_name}_origin.pt" ) 
    return model

MNIST = torchvision.datasets.MNIST(DATA_DIR,
                                   train=True,
                                   transform=None,
                                   target_transform=None, download=True)

ds_len_, ds_ = preprocess_data(MNIST, sigma=None, device=device)
ds_len_, pertubed_ds_ = preprocess_data(MNIST, sigma=[25.0,30.0,40.0], device=device, train=True)
print(type(ds_))
    
sigma = [None, 1e-7, 50.0, 75.0, 100.0]
loaders = [(key,DataLoader(preprocess_data(MNIST, sigma=key, device=device, train=False)[1], batch_size=12000)) for key in sigma]
evaluation = {
    "ode": {
        
    },
    "cnn": {

    }
}
for k in sigma:
    evaluation["ode"].update({k: []})
    evaluation["cnn"].update({k: []})
for i in range(5):
    cnn_model = main(ds_len_,ds_, pertubed_ds_, device=device, model_type="cnn", data_name=f"mnist_origin",batch_size=BATCH_SIZE, epochs=EPOCHS, train_num=TRAIN_NUM, valid_num=VALID_NUM, test_num=TEST_NUM, result_dir=RESULT_DIR, parallel=PARALLEL) 
    ode_model = main(ds_len_,ds_, pertubed_ds_, device=device, model_type="ode", data_name=f"mnist_origin",batch_size=BATCH_SIZE, epochs=EPOCHS, train_num=TRAIN_NUM, valid_num=VALID_NUM, test_num=TEST_NUM, result_dir=RESULT_DIR, parallel=PARALLEL) 
    for k,l in loaders:
        if isinstance(cnn_model, nn.DataParallel): cnn_model = cnn_model.module
        if isinstance(ode_model, nn.DataParallel): ode_model = ode_model.module
        _, cnn_acc = cnn_model.evaluate(l) 
        _, ode_acc = ode_model.evaluate(l) 
        
        print(f"CNNs for {k}-gaussian-pertubed MNIST = {cnn_acc}")
        print(f"ODEs for {k}-gaussian-pertubed MNIST = {ode_acc}")
        

        evaluation["ode"][k].append(ode_acc)
        evaluation["cnn"][k].append(cnn_acc)

with open('./result/evaluate.json', 'w') as fp:
    json.dump(evaluation, fp)
