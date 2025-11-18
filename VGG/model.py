import torch 
from torch import nn
from torchinfo import summary
class VGG(nn.Module):
    
    def __init__(self):
        super(VGG, self).__init__()

        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride = 2)            
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride = 2)  
        )
        self.b3 = nn.Sequential(            
            nn.Conv2d(in_channels=128, out_channels=256,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride = 2) 
        )
        self.b4 = nn.Sequential(            
            nn.Conv2d(in_channels=256, out_channels=512,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride = 2) 
        )
        self.b5 = nn.Sequential(            
            nn.Conv2d(in_channels=512, out_channels=512,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride = 2) 
        )
        self.b6 = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p = 0.4),
            nn.Linear(7*7*512, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Linear(4096,10)
        )

        for m in self.modules():
            if(isinstance(m, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.b6(x)

        return x

if __name__ == "__main__":
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.backends.mps.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("cpu")
    model = VGG().to(device)
    summary(model, (1,1,224,224))