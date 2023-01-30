import torch
import torch.nn.functional as F
import numpy as np

NUM_INPUT_CHANNELS = 1
NUM_HIDDEN_CHANNELS = 64
KERNEL_SIZE = 3
NUM_CONV_LAYERS = 4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_params(num_out):
  meta_parameters = {}
  
  # construct feature extractor
  in_channels = NUM_INPUT_CHANNELS
  for i in range(NUM_CONV_LAYERS):
      meta_parameters[f'conv{i}'] = torch.nn.init.xavier_uniform_(
          torch.empty(NUM_HIDDEN_CHANNELS,in_channels,KERNEL_SIZE,KERNEL_SIZE,requires_grad=True,device=DEVICE))
      meta_parameters[f'b{i}'] = torch.nn.init.zeros_(
          torch.empty(NUM_HIDDEN_CHANNELS,requires_grad=True,device=DEVICE))
      in_channels = NUM_HIDDEN_CHANNELS

  # construct linear head layer
  meta_parameters[f'w{NUM_CONV_LAYERS}'] = torch.nn.init.xavier_uniform_(
      torch.empty(num_out,NUM_HIDDEN_CHANNELS,requires_grad=True,device=DEVICE))
  meta_parameters[f'b{NUM_CONV_LAYERS}'] = torch.nn.init.zeros_(
      torch.empty(num_out,requires_grad=True,device=DEVICE))
  
  return meta_parameters

class MAML:
    def __init__(self,num_out, num_inner_steps, outer_lr=0.001, inner_lr =0.4,learn_inner_lrs=True):
        self.params = get_params(num_out)
        self.num_out = num_out
        self.num_inner_steps = num_inner_steps
        self.outer_lr =outer_lr
        self.inner_lrs = {k: torch.tensor(inner_lr, requires_grad=learn_inner_lrs) for k in self.params.keys()}
        self.optimizer = torch.optim.Adam(list(self.params.values())+list(self.inner_lrs.values()),lr=self.outer_lr)
        self.epochs = 15000
        self.val_iter = 10
        self.best_acc = 0
        
    def forward(self, x, parameters):
        for i in range(NUM_CONV_LAYERS):
            x = F.conv2d(input=x,weight=parameters[f'conv{i}'],bias=parameters[f'b{i}'],stride=1,padding='same')
            x = F.batch_norm(x, None, None, training=True)
            x = F.relu(x)
        x = torch.mean(x, dim=[2, 3])
        out = F.linear(input=x,weight=parameters[f'w{NUM_CONV_LAYERS}'],bias=parameters[f'b{NUM_CONV_LAYERS}'])
        return out

    def adapt(self,xs,ys, adapt=False):
        params = {k: torch.clone(v) for k, v in self.params.items()}

        if not adapt:
            return params

        for i in range(self.num_inner_steps):
            pred = self.forward(xs,params)
            loss = F.cross_entropy(pred,ys.type(torch.LongTensor).to(DEVICE))
            grads = torch.autograd.grad(loss, params.values(), create_graph=True)

            for key,grad in zip(params.keys(),grads):
                params[key] = params[key] - self.inner_lrs[key]*grad

        return params  
    
    def train(self,train_batch, val_batch, adapt=True, is_train=True):
        for epoch in range(self.epochs):
            loss, acc = self.train_one_epoch(train_batch, adapt, is_train)
            if (epoch+1)%self.val_iter==0:
                print(f'Epoch {epoch+1}/{self.epochs}:- Loss: {loss:.2f} Acc: {acc:.2f}')
                val_loss, val_acc = self.train_one_epoch(val_batch,adapt= False, is_train=False)
                print(f'Validation Loss: {val_loss:.2f} Acc: {val_acc:.2f}')
                if self.best_acc<val_acc:
                    self.best_acc = val_acc
                    self.save_model(f"ckpt.pth")
                    

    def train_one_epoch(self,batches, adapt, is_train):
        epoch_loss = []
        epoch_acc = []

        for batch in batches:
            if is_train:
                self.optimizer.zero_grad()

            xs,ys,xq,yq = batch
            xs=xs.to(DEVICE)
            xq=xq.to(DEVICE)
            ys=ys.to(DEVICE)
            yq=yq.to(DEVICE)
            
            with torch.set_grad_enabled(is_train):
                params = self.adapt(xs,ys, adapt=adapt)
                out = self.forward(xq, params).to(DEVICE)
                loss = F.cross_entropy(out,yq.type(torch.LongTensor))
                acc = self.get_acc(out, yq)
            
            if is_train:
                loss.backward()
                self.optimizer.step()

            epoch_loss.append(loss.item())
            epoch_acc.append(acc)

        return np.mean(epoch_loss), np.mean(epoch_acc)

    def test(self, test_batches):
        loss, acc = self.train_one_epoch(test_batches, is_train=False)
        print(f'Test Loss: {loss:.2f} Acc: {acc:.2f}')

    def save_model(self,file_name):
        torch.save(self.params,file_name)

    def load_model(self,file_name):
        self.params.load_state_dict(torch.load(file_name))

    def get_acc(self,out, yq):
        x = torch.argmax(out, dim=-1) == yq.data
        return torch.mean(x.type(torch.float)).item()