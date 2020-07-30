import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, 1)
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1)
        # in_channels：输入图像通道数，手写数字图像为1，彩色图像为3
        # out_channels：输出通道数，这个等于卷积核的数量
        # kernel_size：卷积核大小
        # stride：步长

        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        # 上个卷积网络的out_channels，就是下一个网络的in_channels，所以这里是20
        # out_channels：卷积核数量50
        self.fc1 = nn.Linear(53 * 53 * 50, 500)
        # 全连接层torch.nn.Linear(in_features, out_features)
        # in_features:输入特征维度，4*4*50是自己算出来的，跟输入图像维度有关
        # out_features；输出特征维度

        self.fc2 = nn.Linear(500, 2)
        # 输出维度10，10分类

    def forward(self, x):
        # print(x.shape)  #手写数字的输入维度，(N,1,28,28), N为batch_size
        x = F.relu(self.conv1(x))  # x = (N,50,24,24)
        x = F.max_pool2d(x, 2, 2)  # x = (N,50,12,12)
        x = F.relu(self.conv2(x))  # x = (N,50,8,8)
        x = F.max_pool2d(x, 2, 2)  # x = (N,50,4,4)
        x = x.view(-1, 53 * 53 * 50)  # x = (N,4*4*50)
        x = F.relu(self.fc1(x))  # x = (N,4*4*50)*(4*4*50, 500)=(N,500)
        x = self.fc2(x)  # x = (N,500)*(500, 10)=(N,10)
        return x  # 带log的softmax分类，每张图片返回10个概率


def train(model, train_loader, optimizer, epoch, log_interval=1):
    model.train()
    running_corrects = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        with torch.autograd.set_grad_enabled(True):
            # torch.autograd.set_grad_enabled梯度管理器，可设置为打开或关闭
            # phase=="train"是True和False，双等号要注意
            outputs = model(data)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, target)
        _, preds = torch.max(outputs, 1)
        # 返回每一行最大的数和索引，prds的位置是索引的位置
        # 也可以preds = outputs.argmax(dim=1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_corrects += torch.sum(preds.view(-1) == target.view(-1)).item()
        if batch_idx % log_interval == 0:
            print("Train Epoch: {} [{}/{} ({:0f}%)]\tLoss: {:.6f}\tACC: {}".format(
                epoch,
                batch_idx * len(data),  # 100*32
                len(train_loader.dataset),  # 60000
                100. * batch_idx / len(train_loader),  # len(train_loader)=60000/32=1875
                loss.item(),
                running_corrects
            ))


def test(model, test_loader, loss):
    model.eval()  # 进入测试模式
    test_loss = 0
    correct = 0
    since = time.time()
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            loss_function = nn.CrossEntropyLoss()
            test_loss += loss_function(output, target).item()  # sum up batch loss
            # reduction='sum'代表batch的每个元素loss累加求和，默认是mean求平均

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

            # print(target.shape) #torch.Size([32])
            # print(pred.shape) #torch.Size([32, 1])
            correct += pred.eq(target.view_as(pred)).sum().item()
            # pred和target的维度不一样
            # pred.eq()相等返回1，不相等返回0，返回的tensor维度(32，1)。

    test_loss /= len(test_loader.dataset)
    loss.append(test_loss)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    end = time.time()
    print(end - since)
