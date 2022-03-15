import torch
import copy
from teacher import teacher_net
from student import student_net
from Distiller import Distiller
from data import train_ds,val_ds
from data import trn_dl,val_dl
from torch_snippets import *



teacher_model, teacher_loss_fn, teacher_optimizer = teacher_net()
student_model, student_loss_fn, student_optimizer = student_net()
student_clone = copy.deepcopy(student_model)
distiller = Distiller(teacher=teacher_model, student=student_model, student_loss_fn=student_loss_fn, student_optimizer=student_optimizer)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_batch(data, model, optimizer, loss_fn):
    model.train()
    ims, targets = data
    optimizer.zero_grad()
    predictions = model(ims)
    batch_loss = loss_fn(predictions, targets)
    batch_loss.backward()
    optimizer.step()
    acc = (torch.max(predictions, 1)[1] == targets).float().mean()
    return batch_loss.item(),acc

@torch.no_grad()
def validate_batch(model, data, loss_fn):
    model.eval()
    ims, labels = data
    _preds = model(ims)
    loss = loss_fn(_preds, labels)
    acc = (torch.max(_preds, 1)[1] == labels).float().mean()
    return loss.item(), acc