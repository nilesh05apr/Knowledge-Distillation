import os
import torch
import torch.nn as nn
import torchvision
import tarfile
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split, Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import confusion_matrix
import seaborn as sns


class Distiller(nn.Module):
    def __init__(self, student, teacher, student_loss_fn, student_optimizer):
        super(Distiller, self).__init__()
        self.teacher = teacher
        self.student = student
        self.student_loss_fn = student_loss_fn
        self.student_optimizer = student_optimizer

    def train_step(self, data, alpha, temp):
        ims, targets = data
        self.alpha = alpha
        self.teacher.eval()
        teacher_predictions = self.teacher(ims)
        self.student_optimizer.zero_grad()
        self.student.train()
        student_predictions = self.student(ims)
        student_loss = self.student_loss_fn(student_predictions, targets)
        d_loss_fn = nn.KLDivLoss()
        distillation_loss = d_loss_fn(nn.functional.softmax(teacher_predictions/temp),
                                      nn.functional.softmax(student_predictions/temp),
                                      ).detach()
        loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        loss.backward()
        self.student_optimizer.step()
        acc = (torch.max(student_predictions, 1)[1] == targets).float().mean()
        return loss.item(),acc

    @torch.no_grad()
    def test_step(self, data):
        self.student.eval()
        ims, targets = data
        y_predictions = self.student(ims)
        student_loss = self.student_loss_fn(y_predictions, targets)
        acc = (torch.max(y_predictions, 1)[1] == targets).float().mean()
        return student_loss.item(), acc