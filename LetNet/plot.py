import torch
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch.utils.data as Data

train_data = FashionMNIST(root='./LetNet/data',
                          train = True,
                          transform=transforms.Compose([transforms.Resize(size=224), transforms.ToTensor()]),
                          download=True)

train_loader = Data.DataLoader(dataset=train_data,
                               batch_size=64,
                               shuffle=True,
                               num_workers=0)


for step, (b_x, b_y) in enumerate(train_loader):
    if step>0:
        break

    batch_x = b_x.squeeze().numpy()
    batch_y = b_y.numpy()
    class_labels = train_data.classes

print(train_data)

plt.figure(figsize=(12,5))
for i in np.arange(len(batch_y)):
    plt.subplot(4, 16, i+1)
    plt.imshow(batch_x[i,:,:], cmap = plt.cm.gray)
    plt.title(class_labels[batch_y[i]],size = 8)
    plt.axis("off")
    plt.subplots_adjust(wspace=0.05)
plt.show()

