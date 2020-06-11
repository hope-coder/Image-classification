import os
import time

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torchvision
from sklearn import svm, preprocessing
from torch import nn
from torchvision import datasets, transforms
import CNN
import resnet

print("Torchvision Version: ", torchvision.__version__)

torch.manual_seed(53113)  # cpu随机种子
"""
    首先对参数进行初始化
"""
data_dir = "./Dog_Cat"
batch_size = 32  # 每次梯度降的的数量
input_size = 224  # 输入大小
device = torch.device("cpu")  # 没啥用，反正我都没下cuda
lr = 0.001  # 学习率：影响较大
momentum = 0.9
model = CNN.Net()  # 模型初始化
function = "resnet"  # 训练方法
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)  # 定义优化器
trainset = datasets.ImageFolder(os.path.join(data_dir, "train"),
                                transforms.Compose([
                                    transforms.RandomResizedCrop(input_size),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

                                ]))

testset = datasets.ImageFolder(os.path.join(data_dir, "val"),
                               transforms.Compose([
                                   transforms.Resize(input_size),
                                   transforms.CenterCrop(input_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                               ]))

if function == 'resnet':
    model_name = "resnet"
    num_classes = 2
    feature_extract = True
    model, input_size = resnet.initialize_model('resnet', 2, False, use_pretrained=False)
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)  # 定义优化器
    criterion = nn.CrossEntropyLoss()  # 定义损失函数
    print(model)
    data_transforms = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    print("Initializing Datasets and Dataloaders...")

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0) for x in
        ['train', 'val']}
    model, acc = resnet.train_model(model, dataloaders_dict, criterion, optimizer_ft, 10)
    x = []
    y = []
    for index, acc in enumerate(acc):
        x.append(index*5)
        y.append(acc)
    plt.plot(x, y)

    plt.xlabel('Batch')
    plt.ylabel('Accuracy on test data')
    plt.title('Resnet performance')
    plt.legend()
    plt.show()
if function == "CNN":
    trainset = datasets.ImageFolder(os.path.join(data_dir, "train"),
                                    transforms.Compose([
                                        transforms.RandomResizedCrop(input_size),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

                                    ]))

    testset = datasets.ImageFolder(os.path.join(data_dir, "val"),
                                   transforms.Compose([
                                       transforms.Resize(input_size),
                                       transforms.CenterCrop(input_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                   ]))

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=0)
    epochs = 200
    epoch_list = range(0, 200, 5)

    loss = []
    t1 = time.time()
    for epoch in range(1, epochs + 1):
        CNN.train(model, train_loader, optimizer, epoch)
        t3 = time.time()
        print(t3 - t1)
        CNN.test(model, test_loader, loss)
        if epoch % 20 == 0:
            t3 = time.time()
            print(t3 - t1)
            CNN.test(model, train_loader, loss)

    t2 = time.time()
    print(t2 - t1)
    plt.plot(epoch_list, loss)

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy on test data')
    plt.title('Comparing model performance')
    plt.legend()
    plt.show()