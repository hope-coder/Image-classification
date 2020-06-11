import numpy as np
import torchvision
from torchvision import datasets, transforms, models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
import os
import copy


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    if model_name == "resnet":
        model_ft = models.resnet18(pretrained=use_pretrained)

        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features

        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    return model_ft, input_size


def train_model(model, dataloaders, criterion, optimizer, num_epochs=5, log_interval=5):
    since = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        running_loss = 0.
        running_corrects = 0.
        model.train()
        phase = 'train'
        for batch_id, (inputs, labels) in enumerate(dataloaders[phase]):
            with torch.autograd.set_grad_enabled(True):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_id % log_interval == 0:
                print("Train Epoch: {} [{}/{} ({:0f}%)]\tLoss: {:.6f}\tAcc: {}/{}".format(
                    epoch,
                    batch_id * 32,
                    len(dataloaders['train'].dataset),
                    100. * batch_id / len(dataloaders['train']),
                    loss.item(),
                    int(running_corrects),
                    batch_id * 32
                ))
                epoch_loss, epoch_acc = test_model(model, dataloaders, criterion, epoch)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                val_acc_history.append(epoch_acc)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds.view(-1) == labels.view(-1)).item()

        epoch_loss = running_loss / len(dataloaders[phase].dataset)
        epoch_acc = running_corrects / len(dataloaders[phase].dataset)
        print("{} Loss: {} Acc: {}".format(phase, epoch_loss, epoch_acc))
        print()
    time_elapsed = time.time() - since
    print("Training compete in {}m   {}s".format(time_elapsed // 60, time_elapsed % 60))
    print("Best val Acc: {}".format(best_acc))

    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def test_model(model, dataloaders, criterion, epoch):
    best_model_wts = copy.deepcopy(model.state_dict())
    running_loss = 0.
    running_corrects = 0.
    model.eval()
    since = time.time()
    for inputs, labels in dataloaders['val']:
        with torch.autograd.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds.view(-1) == labels.view(-1)).item()

    epoch_loss = running_loss / len(dataloaders['val'].dataset)
    epoch_acc = running_corrects / len(dataloaders['val'].dataset)
    time_elapsed = time.time() - since
    print("Training compete in {}m   {}s".format(time_elapsed // 60, time_elapsed % 60))
    print("{} Loss: {} Acc: {}".format('val', epoch_loss, epoch_acc))
    model.load_state_dict(best_model_wts)
    return epoch_loss, epoch_acc
