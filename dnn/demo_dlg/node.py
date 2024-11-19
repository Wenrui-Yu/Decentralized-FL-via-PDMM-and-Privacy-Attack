import numpy as np
import random
import math

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.autograd as autograd
from torch.utils.data import Dataset

import collections
import copy
from PdmmSGD import PdmmSGD

class node(object):
    def __init__(self,neighbors, dataset, model,device):
        self.neighbor=np.mat(neighbors)
        self.neighbor=self.neighbor.A
        self.dataloader=dataset

        self.model=model
        self.model.train()
        self.model.to(device)

        self.dual=[[] for i in range(len(self.neighbor[0]))]
        self.y_old=[[] for i in range(len(self.neighbor[0]))]
        worker_state_dict = self.model.state_dict()
        worker_state_dict = {k: v for k, v in worker_state_dict.items() if
                             'running_mean' not in k and 'running_var' not in k and 'num_batches_tracked' not in k}
        weight_keys =list(worker_state_dict.keys())
        for param_tensor in weight_keys:
            for n in range(len(self.neighbor[0])):
                self.dual[n].append(torch.zeros(worker_state_dict[param_tensor].size(), device=device))
                self.y_old[n].append(torch.zeros(worker_state_dict[param_tensor].size(), device=device))
        self.optimizer=PdmmSGD(self.model,self.neighbor,self.dual)

    def init_dual(self,init_d):
        self.dual=init_d.copy()
        worker_state_dict = self.model.state_dict()
        worker_state_dict = {k: v for k, v in worker_state_dict.items() if
                             'running_mean' not in k and 'running_var' not in k and 'num_batches_tracked' not in k}
        weight_keys =list(worker_state_dict.keys())
        i=0
        for param_tensor in weight_keys:
            for n in range(len(self.neighbor[0])):
                self.dual[n][i]=self.dual[n][i]*abs(self.neighbor[0][n])
            i+=1

    def active_update(self,device,epo,theta=1,ro=0.4):

        for e in range(epo):   
            img,label=next(iter(self.dataloader))
            if img.ndim == 3:
                img = img[:, None, :, :]
            else:
                img=img.permute(0,3,1,2)
                label = label.type(torch.LongTensor)
            img,label=img.to(device),label.to(device)
            self.optimizer.zero_grad()
            output=self.model(img)
            loss=F.cross_entropy(output,label)

            loss.backward()
            self.optimizer.update(self.dual)
            self.optimizer.step()

        yij=[[] for i in range(len(self.neighbor[0]))]
        worker_state_dict = self.model.state_dict()
        weight_keys =list(worker_state_dict.keys())
        i=0
        for param_tensor in weight_keys:
            for n in range(len(self.neighbor[0])):
                yij[n].append(self.dual[n][i]+2*ro*self.neighbor[0][n]*worker_state_dict[param_tensor])
            i=i+1

        assert math.isnan(loss.item())==False,'NaN'
        return loss.item(),yij

    def second_update(self,device,yij,index,j,theta=1,ro=0.4):
        self.dual[index]=yij[j]