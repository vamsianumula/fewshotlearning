import numpy as np
import torch
from torchvision import models

class ProtoNet(torch.nn.Module):
  def __init__(self):
    super().__init__()
    model = models.resnet18(pretrained=True)
    self.conv_net = torch.nn.Sequential(*list(model.children())[:-1])
  
  def forward(self, xs, ys, xq):
    x = self.conv_net(xs)
    N = xq.shape[0]

    prot = torch.stack([x[ys==i].mean(0) for i in range(N)]).detach()
    pred = self.conv_net(xq)
    dist = torch.zeros((pred.shape[0], prot.shape[0]))
    
    for i in range(N):
        t = torch.zeros(prot.shape[0])
        for j in range(prot.shape[0]):
            t[j] = torch.dist(pred[i],prot[j],2)
        dist[i] = -t
    return dist

class FSL_ProtoNet:
  def __init__(self,model, config):
    self.batch_size = config["batch_size"]
    self.n = config["n_way"]
    self.n_s = config["num_support"]
    self.n_q = config["num_query"]
    self.lr = config["learning_rate"]
    self.epochs = config["epochs"]
    self.val_iter= config["val_iter"]
    self.model = model
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["learning_rate"])
    self.loss_func = torch.nn.CrossEntropyLoss()
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.best_acc = 0

  def train(self,train_batch, val_batch):
    self.model.to(self.device)
    for epoch in range(self.epochs):
      self.model.train(True)
      loss, acc = self.train_one_epoch(train_batch, is_train=True)

      if (epoch+1)%self.val_iter==0:
        print(f'Epoch {epoch+1}/{self.epochs}:- Loss: {loss:.2f} Acc: {acc:.2f}')
        self.model.train(False)
        val_loss, val_acc = self.train_one_epoch(val_batch, False)
        print(f'Validation Loss: {val_loss:.2f} Acc: {val_acc:.2f}')
        if self.best_acc<val_acc:
            self.best_acc = val_acc
            self.save_model(f"ckpt.pth")
  
  def train_one_epoch(self,batches, is_train=True):
    epoch_loss = []
    epoch_acc = []

    for batch in batches:
      if is_train:
        self.optimizer.zero_grad()

      xs,ys,xq,yq = batch
      xs=xs.to(self.device)
      xq=xq.to(self.device)
      ys=ys.to(self.device)
      yq=yq.to(self.device)

      out = self.model(xs,ys,xq).to(self.device)
      loss = self.loss_func(out, yq)
      acc = self.get_acc(out, yq)
      
      if is_train:
        loss.backward()
        self.optimizer.step()

      epoch_loss.append(loss.item())
      epoch_acc.append(acc)

    return np.mean(epoch_loss), np.mean(epoch_acc)

  def test(self, test_batches):
    self.model.eval()
    loss, acc = self.train_one_epoch(test_batches, is_train=False)
    print(f'Test Loss: {loss:.2f} Acc: {acc:.2f}')
    
  def save_model(self,file_name):
    torch.save(self.model.state_dict(),file_name)
  
  def load_model(self,file_name):
    self.model.load_state_dict(torch.load(file_name))
  
  def get_acc(self,out, yq):
    x = torch.argmax(out, dim=-1) == yq.data
    return torch.mean(x.type(torch.float)).item()