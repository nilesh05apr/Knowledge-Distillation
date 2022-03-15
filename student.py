import os
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torchsummary import summary


device = 'cuda' if torch.cuda.is_available() else 'cpu'



def student_net():
    model = nn.Sequential(

        #Layer 1
        nn.Conv2d(3, 64, kernel_size=5, stride=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        #nn.BatchNorm2d(64),
        
        #Layer2
        nn.Conv2d(64, 128, kernel_size=3, stride=2),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.BatchNorm2d(128),
        nn.Dropout(0.2),

        #Layer3
        nn.Conv2d(128, 264, kernel_size=3, stride=2),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.BatchNorm2d(264),
        
        #Layer4
        nn.Flatten(),
        nn.Linear(9504, 512),
        nn.ReLU(),
        
        #Layer5
        nn.Linear(512, 64),
        nn.ReLU(),
        nn.Dropout(0.25),

        #Output Layer
        nn.Linear(64, 10)
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    return model.to(device), loss_fn, optimizer


stu_mod, stu_loss, stu_opt = student_net()
summary(stu_mod, (3, 224, 224))
print(device)
