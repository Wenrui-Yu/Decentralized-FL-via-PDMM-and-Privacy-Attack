# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
import numpy as np

class PdmmSGD(Optimizer):
    def __init__(self, model,neighbor,dual, mu=20,rho=0.4,clip_value=1):
        self.mu = mu
        self.rho =rho
        self.lr = 1 / self.mu
        super(PdmmSGD, self).__init__(model.parameters(), {})
        self.neighbor=neighbor
        self.dual=dual
        self.clip_value=clip_value

    def update(self,dual):
        self.dual=dual

    def step(self, closure=False): 
        for param_group in self.param_groups:
            params = param_group['params']
            i=0
            for param in params:
                penalty=torch.zeros_like(param.data)
                for j in range(len(self.neighbor[0])):
                    if self.neighbor[0][j]!=0:
                        penalty=penalty+self.neighbor[0][j]*self.dual[j][i]
                pe=(penalty+self.rho*np.count_nonzero(self.neighbor)*param.data)
                param.data = param.data - 1/self.mu*(param.grad+pe)
                # param.data = (self.mu * param.data - (param.grad + penalty).clamp_(-self.clip_value,
                #                                                                    self.clip_value)) / (
                #                      self.mu + self.rho * np.count_nonzero(self.neighbor))
                i=i+1


