import os
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torchsummary import summary




device = 'cuda' if torch.cuda.is_available() else 'cpu'
large_net = models.resnet18(pretrained=True).to(device)
summary(large_net, (3, 160, 160))

def teacher_net():
    model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False
    
    model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))

    model.fc = nn.Sequential(
        nn.Flatten(),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 10),
    )
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return model.to(device), loss_fn, optimizer