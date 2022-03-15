import torch
import copy
from teacher import teacher_net
from student import student_net
from Distiller import Distiller
#from data import train_ds,val_ds
from data import trn_dl,val_dl
from utils import teacher_model,train_batch,validate_batch,device,teacher_loss_fn,teacher_optimizer
from torch_snippets import *





tch_bs_trn_acc = []
tch_bs_val_acc = []


n_epochs = 15
log = Report(n_epochs)
for epoch in range(n_epochs):
    N = len(trn_dl)
    for ix, data in enumerate(trn_dl):
        train_loss,train_acc = train_batch(data, teacher_model, teacher_optimizer, teacher_loss_fn)
        tch_bs_trn_acc.append(train_acc)
        log.record(epoch+(ix+1)/N, trn_loss=train_loss, trn_acc=train_acc, end='\r')
    
    N = len(val_dl)
    for ix, data in enumerate(val_dl):
        validation_loss, accuracy = validate_batch(teacher_model, data, teacher_loss_fn)
        tch_bs_val_acc.append(accuracy)
        log.record(epoch+(ix+1)/N, val_loss=validation_loss, val_acc=accuracy, end='\r')
    
    log.report_avgs(epoch+1)
log.plot_epochs(['trn_acc', 'val_acc'])