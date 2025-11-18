import torch
from torch import nn
from torchinfo import summary
import torch.nn.functional as F

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4):
        super(InceptionBlock, self).__init__()
        self.p1 = nn.Conv2d(in_channels = in_channels, out_channels=c1, kernel_size=1)
        self.p2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=c2[0], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=c2[0],out_channels=c2[1],kernel_size=3, padding=1)
        )
        self.p3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=c3[0],kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=c3[0],out_channels=c3[1],kernel_size=5, padding=2)
        )
        self.p4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
            nn.Conv2d(in_channels=in_channels, out_channels=c4, kernel_size=1)
        )

    def forward(self,x):
        o1 = F.relu(self.p1(x))
        o2 = F.relu(self.p2(x))
        o3 = F.relu(self.p3(x))
        o4 = F.relu(self.p4(x))

        return torch.concat((o1,o2,o3,o4), dim=1)


class GoogLeNet(nn.Module):

    def __init__(self):
        super(GoogLeNet, self).__init__()

        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2,padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2,padding=1)
        )

        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.inception1 = InceptionBlock(192,64,[96,128],[16,32],32)

        self.b3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception2 = InceptionBlock(256, 128, [128, 192], [32, 96], 64)
        self.inception3 = InceptionBlock(480, 192, [96,208],[16,48],64)
        self.inception4 = InceptionBlock(512, 160,[112,224],[24,64],64)
        self.inception5 = InceptionBlock(512, 128, [128, 256],[24,64],64)
        self.inception6 = InceptionBlock(512, 112, [128,288],[32,64],64)
        self.inception7 = InceptionBlock(528, 256, [160,320],[32,128],128)
        self.inception8 = InceptionBlock(832, 256, [160,320],[32,128],128)
        self.inception9 = InceptionBlock(832, 384,[192,384],[48,128],128)

        self.b4 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(1024, 10)
        )
        for m in self.modules():
            if(isinstance(m, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.inception1(x)
        x = self.b3(x)
        x = self.inception2(x)
        x = self.inception3(x)
        x = self.inception4(x)
        x = self.inception5(x)
        x = self.inception6(x)
        x = self.inception7(x)
        x = self.b3(x)
        x = self.inception8(x)
        x = self.inception9(x)

        x = self.b4(x)

        return x

if __name__ == "__main__":
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.backends.mps.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = GoogLeNet().to(device)
    summary(model, (1,3,224,224))