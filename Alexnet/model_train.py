import torch
from torch import nn
from torchvision.datasets import CIFAR10
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch.utils.data as Data
from model import AlexNet
from torchinfo import summary
import copy
import time

def train_val_data_process():
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                    std=[0.2023, 0.1994, 0.2010])
    train_data =CIFAR10(root='./AlexNet/data',
                        train = True,
                        transform=transforms.Compose([transforms.Resize(size=227), transforms.ToTensor(),normalize]),
                        download=True)
    
    train_data, val_data = Data.random_split(train_data, [round(0.8*len(train_data)), round(0.2*len(train_data))])

    train_dataloader = Data.DataLoader(dataset=train_data,
                                       batch_size = 64,
                                       shuffle=True,
                                       num_workers=0)
    val_dataloader = Data.DataLoader(dataset=val_data,
                                       batch_size = 64,
                                       shuffle=True,
                                       num_workers=0)
    
    return train_dataloader, val_dataloader

def train_model_process(model, train_dataloader, val_dataloader, epochs):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.backends.mps.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("cpu")
    
    model = model.to(device)

    #* Setting the optimizer and learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    #* Setting the loss function
    loss = nn.CrossEntropyLoss()

    #* Saving the best model 
    best_model_wts = copy.deepcopy(model.state_dict())

    #* Initialing the test value
    best_acc = 0.0

    train_loss_all = []
    train_acc_all = []
    val_loss_all = []
    val_acc_all = []

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        print(f"The epoch {epoch} / {epochs}: ")

        print("-"*15)

        train_loss = 0
        train_acc = 0
        train_num = 0
        val_loss = 0
        val_acc = 0
        val_num = 0

        for step, (b_x, b_y) in enumerate(train_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            #* update the weight 
            model.train()
            
            output = model(b_x)
            pre_lab = torch.argmax(output, dim = 1)

            data_loss = loss(output, b_y)

            optimizer.zero_grad()
            data_loss.backward()
            optimizer.step()

            #* acculate the average loss and acc of each train
            train_loss += data_loss.item()*b_x.size(0)
            train_acc += torch.sum(pre_lab == b_y.data)

            train_num += b_x.size(0)

        for step, (b_x, b_y) in enumerate (val_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            model.eval()
            output = model(b_x)

            pre_lab = torch.argmax(output, dim = 1)

            data_loss = loss(output, b_y)

            val_loss += data_loss.item()*b_x.size(0)
            val_acc += torch.sum(pre_lab == b_y.data)
            val_num += b_x.size(0)
        
        #* Save each epoch loss and acc 
        train_loss_all.append(train_loss/train_num)
        train_acc_all.append(train_acc.item()/train_num)

        val_loss_all.append(val_loss/val_num)
        val_acc_all.append(val_acc.item()/val_num)
        print(f"{epoch} Train Loss: {train_loss_all[-1]:.4f} Train Acc: {train_acc_all[-1]:.4f}")
        print(f"{epoch} Val Loss: {val_loss_all[-1]:.4f} Val Acc: {val_acc_all[-1]:.4f}")

        if val_acc_all[-1]>best_acc:
            best_acc = val_acc_all[-1]

            best_model_wts = copy.deepcopy(model.state_dict())
        time_use = time.time()-epoch_start_time
        print(f"Time cost in Epoch {epoch} : {time_use}s")

    #* Save the best model

    torch.save(best_model_wts, 'AlexNet/best_model.pth')

    train_process = pd.DataFrame(data={"epoch":range(epochs),
                                        "train_loss":train_loss_all,
                                        "train_acc":train_loss_all,
                                        "val_loss":val_loss_all,
                                        "val_acc":val_acc_all})
    
    return train_process

def matplot_acc_loss(train_process):
    plt.figure(figsize=(12,4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process["epoch"], train_process.train_loss_all, 'ro-', label = "train loss")
    plt.plot(train_process["epoch"], train_process.val_loss_all, 'bs-', label = "val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")


    plt.subplot(1, 2, 2)
    plt.plot(train_process["epoch"], train_process.train_acc_all, 'ro-', label = "train acc")
    plt.plot(train_process["epoch"], train_process.val_acc_all, 'bs-', label = "val acc")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.show()

if __name__ == "__main__":
    AlexNet = AlexNet()
    train_dataloader, val_dataloader = train_val_data_process()

    train_process = train_model_process(AlexNet, train_dataloader, val_dataloader, 30)
    matplot_acc_loss(train_process)