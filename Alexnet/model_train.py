import torch
from torch import nn
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch.utils.data as Data
from model import VGG  # 假设你的 model.py 在同一个文件夹
from torchinfo import summary
import copy
import time
import os # [Refactor 1] 导入 os 模块，用于创建文件夹

def train_val_data_process():
    
    # [Refactor 2] 为 FashionMNIST (1通道) 添加标准的均值和标准差
    # 这会使 VGG 这种深度网络的训练更加稳定
    FASHION_MNIST_MEAN = (0.2860,)
    FASHION_MNIST_STD = (0.3530,)

    data_transform = transforms.Compose([
        transforms.Resize(size=224),
        transforms.ToTensor(),
        transforms.Normalize(FASHION_MNIST_MEAN, FASHION_MNIST_STD) # <-- [Refactor 2] 添加归一化
    ])

    train_data = FashionMNIST(root='./VGG/data',
                            train=True,
                            transform=data_transform,
                            download=True)
    
    # 将 train_data 分为训练集和验证集
    train_data, val_data = Data.random_split(train_data, [round(0.8*len(train_data)), round(0.2*len(train_data))])

    train_dataloader = Data.DataLoader(dataset=train_data,
                                       batch_size=512,
                                       shuffle=True,
                                       num_workers=12,
                                       pin_memory=True) # <-- [Refactor 3] 加速 CUDA 数据传输

    val_dataloader = Data.DataLoader(dataset=val_data,
                                       batch_size=512,
                                       shuffle=True, # shuffle 验证集不是必须的，但也没问题
                                       num_workers=12,
                                       pin_memory=True) # <-- [Refactor 3] 加速 CUDA 数据传输
    
    return train_dataloader, val_dataloader

def train_model_process(model, train_dataloader, val_dataloader, epochs):
    
    # 优先使用 MPS (Mac), 其次 CUDA (Nvidia), 最后 CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss() # <-- [Refactor 4] 重命名为 loss_fn 避免与变量名冲突
    
    # 将模型和损失函数移动到目标设备
    model = model.to(device)
    loss_fn = loss_fn.to(device)

    # 最佳模型的权重
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # 存储训练历史
    train_loss_all = []
    val_loss_all = []
    train_acc_all = []
    val_acc_all = []

    # 记录总训练开始时间
    total_train_start_time = time.time()

    for epoch in range(1, epochs + 1):
        
        epoch_start_time = time.time() # 记录 epoch 开始时间

        print(f'The {epoch}/{epochs} Epoch')
        print("-" * 10)

        # --- 训练阶段 ---
        model.train() # <-- [Refactor 5] 将 .train() 移到循环外部，无需每批都设置

        train_loss = 0
        train_acc = 0
        train_num = 0

        for step, (b_x, b_y) in enumerate(train_dataloader):
            # 将数据移动到设备
            b_x, b_y = b_x.to(device), b_y.to(device)

            output = model(b_x)
            data_loss = loss_fn(output, b_y) # 使用 loss_fn
            pre_lab = torch.argmax(output, dim=1)

            # 反向传播
            optimizer.zero_grad()
            data_loss.backward()
            optimizer.step()

            # 累加统计
            train_loss += data_loss.item() * b_x.size(0)
            train_acc += torch.sum(pre_lab == b_y) # <-- [Refactor 6] 移除 .data，更现代的写法
            train_num += b_x.size(0)

        # --- 验证阶段 ---
        model.eval() # <-- [Refactor 5] 将 .eval() 移到循环外部

        val_loss = 0
        val_acc = 0
        val_num = 0

        with torch.no_grad(): # <-- [Refactor 7] 验证时关闭梯度计算，节省显存和时间
            for step, (b_x, b_y) in enumerate(val_dataloader):
                # 将数据移动到设备
                b_x, b_y = b_x.to(device), b_y.to(device)

                output = model(b_x)
                data_loss = loss_fn(output, b_y)
                pre_lab = torch.argmax(output, dim=1)

                # 累加统计
                val_loss += data_loss.item() * b_x.size(0)
                val_acc += torch.sum(pre_lab == b_y) # <-- [Refactor 6] 移除 .data
                val_num += b_x.size(0)
        
        # --- 记录和打印 Epoch 结果 ---
        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_acc.item() / train_num) # .item() 用于从 0 维张量中取 Python 数字
        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_acc.item() / val_num)
        
        print(f"{epoch} Train Loss: {train_loss_all[-1]:.4f} Train Acc: {train_acc_all[-1]:.4f}")
        print(f"{epoch} Val Loss: {val_loss_all[-1]:.4f} Val Acc: {val_acc_all[-1]:.4f}")

        # 保存最佳模型
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f"Updated best model with Val Acc: {best_acc:.4f}")

        epoch_time_use = time.time() - epoch_start_time
        print(f"Time cost in Epoch {epoch} : {epoch_time_use:.2f}s")
        print("-" * 10)

    total_time_use = time.time() - total_train_start_time
    print(f"Training finished. Total time: {total_time_use // 60:.0f}m {total_time_use % 60:.0f}s")
    print(f"Best Val Acc: {best_acc:.4f}")
    
    # [Refactor 8] 自动创建保存模型的文件夹，防止 FileNotFoundError
    save_dir = 'VGG'
    os.makedirs(save_dir, exist_ok=True)
    torch.save(best_model_wts, os.path.join(save_dir, 'best_model.pth'))
    print(f"Best model saved to {os.path.join(save_dir, 'best_model.pth')}")

    train_process = pd.DataFrame(data={"epoch": range(epochs), # epoch 从 0 到 epochs-1
                                        "train_loss_all": train_loss_all,
                                        "val_loss_all": val_loss_all,
                                        "train_acc_all": train_acc_all,
                                        "val_acc_all": val_acc_all})
    return train_process


def matplot_acc_loss(train_process):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_process.index, train_process.train_loss_all, 'ro-', label="train loss")
    plt.plot(train_process.index, train_process.val_loss_all, 'bs-', label="val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Loss Curve")

    plt.subplot(1, 2, 2)
    plt.plot(train_process.index, train_process.train_acc_all, 'ro-', label="train acc")
    plt.plot(train_process.index, train_process.val_acc_all, 'bs-', label="val acc")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.title("Accuracy Curve")

    plt.show()

if __name__ == "__main__":
    
    # 实例化你的自定义 VGG (1通道输入)
    try:
        vgg_model = VGG()
        
        # 打印模型摘要 (可选，但非常有用)
        # 注意: 需要 batch_size, 通道, H, W
        print("Model Summary:")
        summary(vgg_model, input_size=(1, 1, 224, 224), device="cpu") # 用 (1, 1, ...) 打印摘要

        print("Starting data processing...")
        train_loader, val_loader = train_val_data_process()
        
        print("Starting model training...")
        history = train_model_process(vgg_model, train_loader, val_loader, 20)

        print("Plotting results...")
        matplot_acc_loss(history)
        
    except ImportError:
        print("Error: Could not import 'VGG' from 'model.py'.")
        print("Please ensure 'model.py' is in the same directory and contains a 'VGG' class.")
    except Exception as e:
        print(f"An error occurred: {e}")
        # 打印更详细的追溯信息
        import traceback
        traceback.print_exc()