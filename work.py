import os
import time

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torchvision
from torch import nn
from torchvision import datasets, transforms

import CNN
import resnet

# 打印下框架版本
print("TorchVision Version: ", torchvision.__version__)

# cpu随机种子
torch.manual_seed(53113)


def show(xlabel, ylabel, y):
    """
    绘图函数，将训练的
    :param xlable: x标签名
    :param ylable: y标签名
    :param y: 要绘制的y值
    :return:
    """

    xList = []
    yList = []
    for index, yy in enumerate(y):
        xList.append(index)
        yList.append(yy)
    plt.plot(xList, yList)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title('performance')
    plt.legend()
    plt.show()


"""
    首先对参数进行初始化
"""
data_dir = "./Dog_Cat"  # 设置要处理图片的目录
batch_size = 32  # 每次梯度降的的数量
input_size = 224  # 输入大小（reset默认是224）
device = torch.device("cpu")  # 没啥用，反正我都没下cuda
lr = 0.001  # 学习率：影响较大
momentum = 0.9
model = CNN.Net()  # 模型初始化
function = "CNN"  # 训练方法
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)  # 定义优化器

if function == 'resnet':
    model_name = "resnet"
    num_classes = 2
    feature_extract = True
    model, input_size = resnet.initialize_model('resnet', 2, feature_extract, use_pretrained=True)
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad:
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

    # 训练模型
    model, acc = resnet.train_model(model, dataloaders_dict, criterion, optimizer_ft, 2)

    show('5epoch', 'Accuracy', acc)
    # 测试部分，需要将待测试的图片放在'Dog_Cat/test'文件夹内
    resnet.work_model(model, data_dir, input_size)

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
    epochs = 10
    loss = []
    t1 = time.time()
    for epoch in range(1, epochs + 1):
        CNN.train(model, train_loader, optimizer, epoch)
        t3 = time.time()
        print(t3 - t1)
        if epoch % 2 == 0:
            t3 = time.time()
            print(t3 - t1)
            CNN.test(model, test_loader, loss)
    t2 = time.time()
    print(t2 - t1)
    show('epoch', 'loss', loss)
