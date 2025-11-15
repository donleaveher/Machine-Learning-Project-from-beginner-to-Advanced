import torch
from torch import nn
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import matplotlib.pyplot as plt
import pandas as pd
import torch.utils.data as Data
from model import LeNet
from torchinfo import summary
import copy


def test_val_data_process():
    test_data = FashionMNIST(root='./LetNet/data',
                            train = False,
                            transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]),
                            download=True)
    

    test_dataloader = Data.DataLoader(dataset=test_data,
                                       batch_size = 1,
                                       shuffle=True,
                                       num_workers = 0)
    
    return test_dataloader

def test_model_process(model, test_dataloader):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    model = model.to(device)
    test_acc = 0.0
    test_num = 0
    

    with torch.no_grad():
        for step, (b_x, b_y) in enumerate(test_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            model.eval()
            output = model(b_x)
            pre_lab = torch.argmax(output, dim=1)

            test_acc += torch.sum(pre_lab==b_y.data)
            test_num += b_x.size(0)
        
    
    print(f"The accuracy of testing: {test_acc/test_num}")

if __name__ == "__main__":
    model = LeNet()

    model.load_state_dict(torch.load('LetNet/best_model.pth'))
    test_loader = test_val_data_process()

    test_model_process(model, test_loader)
