import random
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import Omniglot
from torch.utils.data import Sampler, DataLoader

class CustomSampler(Sampler):
  def __init__(self, dataset,config) -> None:
    self.d = {}
    self.n=config["n_way"]
    self.n_s=config["num_support"]
    self.n_q=config["num_query"]
    self.batch_size= config["batch_size"]
    for i,j in enumerate(dataset):
      if j[1] not in self.d:
        self.d[j[1]]=[i]
      else:
        self.d[j[1]].append(i)
    
    del_keys = []
    for i,j in self.d.items():
      if len(j)< self.n_s+self.n_q:
        del_keys.append(i)
    
    for key in del_keys:
      self.d.pop(key)
    
  def __iter__ (self):
    l=[]
    for i in range(self.batch_size):
      labels = random.sample(list(self.d.keys()),self.n)
      x = np.array([random.sample(self.d[j],self.n_s+self.n_q) for j in labels])
      l.append(np.concatenate(x).ravel().tolist())
    return iter(l)

  def __len__(self):
    return self.batch_size

class Dataset:
    def __init__(self, config):
        self.config = config
        self.n=config["n_way"]
        self.n_s=config["num_support"]
        self.n_q=config["num_query"]
        self.batch_size= config["batch_size"]
    
    def shuffle(self,a,b):
        assert len(a)==len(b)
        a = torch.stack(a)
        b = torch.tensor(b,dtype=torch.int)
        p = np.random.permutation(len(a))
        return a[p],b[p].type(torch.LongTensor)
    
    def collate_func(self,batch):
        x_s, y_s, x_q, y_q = [],[],[],[]
        d= {}
        for i,j in batch:
            if j not in d:
                d[j]=[i]
            else:
                d[j].append(i)
        
        for idx, (k,v) in enumerate(d.items()):
            random.shuffle(v)
            y_q+=[idx]*len(v[:self.n_q])
            x_q+=v[:self.n_q]
            x_s+= v[self.n_q:]
            y_s+= [idx]*len(v[self.n_q:])
        
        x_s, y_s = self.shuffle(x_s,y_s)
        x_q, y_q = self.shuffle(x_q, y_q)

        return x_s,y_s,x_q,y_q
    
    def get_data(self,nout=3):
        train_ds = Omniglot(root="./data", download=True,background=True,
                    transform=transforms.Compose([transforms.Grayscale(num_output_channels=nout),
                        transforms.Resize(28),
                        transforms.ToTensor()
                        ]))
        train_size = int(0.8*len(train_ds))
        val_size = int(0.2*len(train_ds))
        train_data, val_data = torch.utils.data.random_split(train_ds, [train_size, val_size])
        test_data = Omniglot(root="./data", download=True,background=False,
                        transform=transforms.Compose([transforms.Grayscale(num_output_channels=nout),
                            transforms.Resize(28),
                            transforms.ToTensor()]))
        return train_data, val_data, test_data

    def get_dataloader(self,data):
        return DataLoader(data, batch_sampler=CustomSampler(data,self.config), collate_fn=self.collate_func)
    
    def view_batch(self, x):
        grid_img = torchvision.utils.make_grid(x)
        plt.imsave('batch.png',grid_img.permute(1, 2, 0).numpy()) 