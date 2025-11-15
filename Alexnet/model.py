import torch 
from torch import nn
from torchinfo import summary
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        self.relu = nn.ReLU()
        self.c1 = nn.Conv2d(in_channels=3,out_channels=96, stride = 4, kernel_size=5)
        self.p1 = nn.AvgPool2d(kernel_size=3, stride = 2)
        self.c2 = nn.Conv2d(in_channels=96,out_channels=256, padding=2, stride=1, kernel_size=5)
        self.p2 = nn.AvgPool2d(kernel_size=3, stride = 2)
        self.c3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        self.c4 = nn.Conv2d(in_channels=384, out_channels=384, padding=1, kernel_size=3)
        self.c5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.p3 = nn.AvgPool2d(kernel_size=3, stride = 2)

        self.fa = nn.Flatten()

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(9216,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096,1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000,10)
        )
    

    def forward(self, x):
        x = self.relu(self.c1(x))
        x = self.p1(x)
        x = self.relu(self.c2(x))
        x = self.p2(x)
        x = self.relu(self.c3(x))
        x = self.relu(self.c4(x))
        x = self.relu(self.c5(x))
        x = self.p3(x)
        x = self.fa(x)

        x = self.classifier(x)
        return x

        
