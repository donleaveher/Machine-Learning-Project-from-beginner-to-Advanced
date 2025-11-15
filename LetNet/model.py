import torch
from torch import nn
from torchinfo import summary

class LeNet(nn.Module):
    def __init__ (self):
        super(LeNet, self).__init__()

        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.sig = nn.Sigmoid()
        self.s1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=0)
        
        self.s2 = nn.AvgPool2d(kernel_size=2, stride = 2)
        self.fa = nn.Flatten()
        self.f6 = nn.Linear(400, 120)
        self.f7 = nn.Linear(120, 84)
        self.f8 = nn.Linear(84,10)
        
    def forward(self, x):
        
        x = self.sig(self.c1(x))
        x = self.s1(x)
        x = self.sig(self.c2(x))
        x = self.s2(x)
        
        x = self.fa(x)
        x = self.f6(x)
        x = self.f7(x)
        x = self.f8(x)
        return x
    
if __name__ == "__main__":
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = LeNet().to(device)
    #* Parameter model, (batch size, channels, H, W)
    summary(model, (1,1,28,28))