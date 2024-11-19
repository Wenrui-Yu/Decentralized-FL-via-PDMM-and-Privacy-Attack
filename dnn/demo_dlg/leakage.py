import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
# import torchvision
import numpy as np
import cv2

def leakage_from_gradients(model_t,model_t2,dummy_data_init,dummy_label_init,gt_data,gt_label):

  dummy_data=dummy_data_init
  dummy_label=dummy_label_init
  optimizer = torch.optim.LBFGS([dummy_data, dummy_label])

  grad_diff_list=[]
  for iters in range(30):
    def closure():

      optimizer.zero_grad()

      dummy_data_proj = dummy_data

      dummy_pred_t = model_t(dummy_data_proj) 
      dummy_loss_t = F.cross_entropy(dummy_pred_t, dummy_label)
      dummy_grad_t = grad(dummy_loss_t, model_t.parameters(), create_graph=True)

      dummy_pred_t2 = model_t2(dummy_data_proj) 
      dummy_loss_t2 = F.cross_entropy(dummy_pred_t2, dummy_label)
      dummy_grad_t2 = grad(dummy_loss_t2, model_t2.parameters(), create_graph=True)

      pred = model_t(gt_data)
      y = F.cross_entropy(pred, gt_label)
      dy_dx = grad(y, model_t.parameters())
      original_dy_dx = list((_.detach().clone() for _ in dy_dx))

      pred2 = model_t2(gt_data)
      y2 = F.cross_entropy(pred2, gt_label)
      dy_dx2 = grad(y2, model_t2.parameters())
      original_dy_dx2 = list((_.detach().clone() for _ in dy_dx2))

      grad_diff = 0
      for gx, gy,gxx,gyy in zip(dummy_grad_t, dummy_grad_t2,original_dy_dx,original_dy_dx2):
        grad_diff += ((gx - gy-gxx+gyy) ** 2).sum()
      grad_diff.backward()

      grad_diff_list.append(grad_diff.detach().cpu().numpy())
      return grad_diff
    
    optimizer.step(closure)
    
  return  dummy_data, dummy_label, grad_diff_list