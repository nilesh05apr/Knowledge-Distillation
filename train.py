import torch
import copy
from teacher import teacher_net
from student import student_net
from Distiller import Distiller
#from data import train_ds,val_ds
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





dist_small_net_trn_acc = []
dist_small_net_val_acc = []

n_epochs = 15
log = Report(n_epochs)
for epoch in range(n_epochs):
    N = len(trn_dl)
    for ix, data in enumerate(trn_dl):
        train_loss,train_acc = distiller.train_step(data, 0.1, 10)
        dist_small_net_trn_acc.append(train_acc)
        log.record(epoch+(ix+1)/N, loss=train_loss, trn_acc=train_acc, end='\r')
    
    N = len(val_dl)
    for ix, data in enumerate(val_dl):
        stu_loss, accuracy = distiller.test_step(data)
        dist_small_net_val_acc.append(accuracy)
        log.record(epoch+(ix+1)/N, val_loss=stu_loss, val_acc=accuracy, end='\r')
    
    log.report_avgs(epoch+1)
log.plot_epochs(['trn_acc','val_acc'])