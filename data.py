import os
import cv2
import numpy as np
import pandas as pd
import tarfile 
import torchvision
import torch
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split, Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
#from utils import device



device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/imagewoof-160.tgz"
download_url(dataset_url, '.')

with tarfile.open('./imagewoof-160.tgz', 'r:gz') as tar: #read file in r mode
  def is_within_directory(directory, target):
      
      abs_directory = os.path.abspath(directory)
      abs_target = os.path.abspath(target)
  
      prefix = os.path.commonprefix([abs_directory, abs_target])
      
      return prefix == abs_directory
  
  def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
  
      for member in tar.getmembers():
          member_path = os.path.join(path, member.name)
          if not is_within_directory(path, member_path):
              raise Exception("Attempted Path Traversal in Tar File")
  
      tar.extractall(path, members, numeric_owner=numeric_owner) 
      
  
  safe_extract(tar, path="./data")


data_dir = './data/imagewoof-160'
print(os.listdir(data_dir))

print(os.listdir('./data/imagewoof-160/train'))
print(len(os.listdir('./data/imagewoof-160/train')))

classes = ['Australian terrier', 'Border terrier', 'Samoyed', 'Beagle', 'Shih-Tzu', 'English foxhound', 'Rhodesian ridgeback', 'Dingo', 'Golden retriever', 'Old English sheepdog']
print(len(classes))



label2target = dict()
target2label = dict()
for ex, i in enumerate(os.listdir('./data/imagewoof-160/train')):
    label2target[i] = ex
    target2label[ex] = i


from glob import glob
trains = []
for k, v in label2target.items():
    trains.extend(glob(f'data/imagewoof-160/train/{k}/*.JPEG'))


trn_tfms = T.Compose([
        T.ToPILImage(),
        T.Resize(224),
        T.ColorJitter(brightness=(0.95, 1.05),
                      contrast=(0.95, 1.05),
                      saturation=(0.95, 1.05),
                      hue=0.05),
        T.RandomAffine(5, translate=(0.01, 0.1)),
        T.ToTensor(),
        T.Normalize(mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2023, 0.1994, 0.2010]),
])

val_tfms = T.Compose([
        T.ToPILImage(),
        T.Resize(224),
        T.ToTensor(),
        T.Normalize(mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2023, 0.1994, 0.2010]),
])



train_data_dir = 'data/imagewoof-160/train'
val_data_dir = 'data/imagewoof-160/val'


train_dir = 'data/imagewoof-160/train'
val_dir = 'data/imagewoof-160/val'


class ImageWoofData(Dataset):
    def __init__(self, folder, transform=None):
        self.fpaths = []
        for k,v in label2target.items():
            self.fpaths.extend(glob(f'{folder}/{k}/*JPEG'))
        self.normalize = transform
        from random import shuffle, seed; seed(10)
        shuffle(self.fpaths)
        self.targets = [label2target[fpath.split('/')[-2]] for fpath in self.fpaths]

    def __len__(self):
        return len(self.fpaths)
    
    def __getitem__(self, ix):
        f = self.fpaths[ix]
        target = self.targets[ix]
        im = (cv2.imread(f)[:, :, ::-1])
        im = cv2.resize(im, (224, 224))
        im = torch.tensor(im/255)
        im = im.permute(2, 0, 1)
        im = self.normalize(im)
        return im.float().to(device), torch.tensor([target])

    def collate_fn(self, batch):
        ims, targets = [], []
        for im, target in batch:
            ims.append(im[None])
            targets.append(target)
        ims = torch.cat(ims).to(device)
        targets = torch.tensor(targets).to(device)
        return ims, targets


train_ds = ImageWoofData(train_data_dir, trn_tfms)
val_ds = ImageWoofData(val_data_dir, val_tfms)


trn_dl = DataLoader(train_ds, batch_size=32, drop_last=True, shuffle=True,collate_fn=train_ds.collate_fn)
val_dl = DataLoader(val_ds, batch_size=32, collate_fn=val_ds.collate_fn, shuffle=True)


