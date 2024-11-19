import pickle
import numpy as np
import cv2
import scipy.io as sio
from collections import OrderedDict
import torch
from model import Net
from scipy.optimize import root
import numpy as np
from leakage import leakage_from_gradients
import os
import random
import copy as cp

class args:
    batchsize = 8
    num_of_clients = 50
    ro=0.4
    theta=1
    dataset='mnist' 
    rho=0.4
    z_std=0
    epo=30

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def label_to_onehot(target, num_classes=10):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target

### training set
with open('dataset_ni{}_N{}.pkl'.format(args.batchsize,args.num_of_clients), 'rb') as f:
    split_datasets = pickle.load(f)
datasets=split_datasets[0][0]
labels=split_datasets[0][1]
for node in range(1,args.num_of_clients):
    datasets=np.concatenate((datasets,split_datasets[node][0]),axis=0)
    labels=np.concatenate((labels,split_datasets[node][1]),axis=0)

for i in range(args.batchsize*args.num_of_clients):
    image = datasets[i,:,:]
    image = 127 * (image + 1)
    image = cv2.resize(image, (32, 32))
    cv2.imwrite('images8/{}.png'.format(i), image)

path='images_ni{}_t20'.format(args.batchsize)
if os.path.exists(path)==False:
    os.mkdir(path)

net=sio.loadmat('network.mat')
AA = net['AA']
A = net['A']

data_size=split_datasets[node][0].size()
# label_size=label_to_onehot(split_datasets[node][1]).size()
label_size = torch.zeros([args.batchsize,10]).size()

for node in range(args.num_of_clients):
# for node in range(11,50):

    for t in range(20,21):
    # for t in range(2,3):

        with open('model_ni{}_N{}_t{}_z{}_e{}.pkl'.format(args.batchsize,args.num_of_clients,t,args.z_std,args.epo), 'rb') as f:
            models_t = pickle.load(f)
        with open('model_ni{}_N{}_t{}_z{}_e{}.pkl'.format(args.batchsize,args.num_of_clients,t+1,args.z_std,args.epo), 'rb') as f:
            models_t2 = pickle.load(f)
        
        w_t=models_t[node]
        w_t2=models_t2[node]

        for i in range(10):
        # for i in labels[node * args.batchsize:(1 + node) * args.batchsize]:
            i=int(i)
            model_t=Net().to(device)
            model_t.load_state_dict(w_t)
            model_t2=Net().to(device)
            model_t2.load_state_dict(w_t2)

            tmp_data= torch.zeros(data_size)
            tmp_label=torch.zeros(label_size)
            tmp_grad_diff=100

            gt_label = torch.zeros([args.batchsize, 10]).to(device)
            for ii in range(args.batchsize):
                gt_label[ii, int(labels[node * args.batchsize + ii])] = 1
            gt_data = torch.tensor(
                datasets[node * args.batchsize:(node + 1) * args.batchsize, :, :].reshape(
                    [args.batchsize, 3, 32, 32])).to(device)

            for rep in range(20):
                dummy_data_in = torch.zeros(data_size)
                dummy_data_in[0,:,:] = torch.randn(data_size[1:])
                dummy_label_in=torch.zeros(label_size)
                dummy_label_in[0,i] = 1
                dummy_data_in = dummy_data_in.to(device).requires_grad_(True)
                dummy_label_in = dummy_label_in.type(torch.float32).to(device)
                dummy_data, dummy_label, grad_diff_list = leakage_from_gradients(model_t,model_t2,dummy_data_in,dummy_label_in,gt_data,gt_label)

                if grad_diff_list[-1]<tmp_grad_diff and grad_diff_list[-1]!=0:
                    tmp_data = dummy_data
                    tmp_label = dummy_label
                    tmp_grad_diff=grad_diff_list[-1]
            
            image = 127*(tmp_data[0,:,:].cpu().detach().numpy()+1)
            image = (image - image.min()) / (image.max() - image.min()) * 256
            image = cv2.resize(image, (32, 32))

            cv2.imwrite('images_ni{}_t20/{}_{}.png'.format(args.batchsize,node,i), image)
            print(node,tmp_grad_diff,i)