import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image
from inversefed.data.loss import Classification, PSNR
import torchvision.transforms as transforms
from torchvision import datasets

num_images = 1
trained_model = True

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
print(torch.cuda.device_count()) 

import inversefed
import pickle
setup = inversefed.utils.system_startup()
defs = inversefed.training_strategy('conservative')

loss_fn, trainloader, validloader =  inversefed.construct_dataloaders('CIFAR100', defs)

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
# import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.autograd as autograd
from torch.utils.data import Subset,Dataset
import torchvision
from torchvision.models.resnet import Bottleneck
from torchvision.models.resnet import BasicBlock

cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):
    def __init__(self, features, num_class=200):
        super().__init__()
        self.features = features

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )
    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)
        return output

def make_layers(cfg, batch_norm=False):
    layers = []
    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue
        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]
        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l
    return nn.Sequential(*layers)

def vgg11_bn():
    return VGG(make_layers(cfg['A'], batch_norm=True))

def vgg13_bn():
    return VGG(make_layers(cfg['B'], batch_norm=True))

def vgg16_bn():
    return VGG(make_layers(cfg['D'], batch_norm=True))

def vgg19_bn():
    return VGG(make_layers(cfg['E'], batch_norm=True))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
 
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=2), 
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=256*6*6,out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=100),
            nn.Softmax()
        )
 
 
    def forward(self,x):
        return self.model(x)

class args:
    batchsize = 1
    num_of_clients = 10
    ro=0.4
    theta=1
    dataset='mnist' 
    rho=0.4
    z_std=0
    epo=1        
    t=20
with open('./model_ni125_N8_t10_z0_iid1_cf100_alex.pkl','rb') as f:
    models_t = pickle.load(f)
with open('./model_ni125_N8_t11_z0_iid1_cf100_alex.pkl','rb') as f:
    models_t2 = pickle.load(f)

for node in range(8):
    w_t = models_t[node]
    w_t2 = models_t2[node]


    # model, _ = inversefed.construct_model(arch, num_classes=100, num_channels=3)
    model=Net()
    model.load_state_dict(w_t)
    model.to(**setup)
    model.eval()
    # model2, _ = inversefed.construct_model(arch, num_classes=100, num_channels=3)
    model2=Net()
    model2.load_state_dict(w_t2)
    model2.to(**setup)
    model2.eval()

    dm = torch.as_tensor(inversefed.consts.cifar10_mean, **setup)[:, None, None]
    ds = torch.as_tensor(inversefed.consts.cifar10_std, **setup)[:, None, None]

    img, label = validloader.dataset[node]
    ground_truth= torch.as_tensor(img.to(**setup))
    ground_truth=ground_truth.unsqueeze(0)
    labels = torch.as_tensor((1,), device=setup['device'])

    model.zero_grad()
    target_loss, _, _ = loss_fn(model(ground_truth), labels)
    input_gradient = torch.autograd.grad(target_loss, model.parameters())
    input_gradient = [grad.detach() for grad in input_gradient]
    model2.zero_grad()
    target_loss2, _, _ = loss_fn(model2(ground_truth), labels)
    input_gradient2 = torch.autograd.grad(target_loss2, model2.parameters())
    input_gradient2 = [grad.detach() for grad in input_gradient2]

    config = dict(signed=True,
                boxed=True,
                cost_fn='sim',
                indices='def',
                weights='equal',
                lr=0.1,
                optim='adam',
                restarts=4,
                max_iterations=2400,
                total_variation=1e-2,
                init='randn',
                filter='none',
                lr_decay=True,
                scoring_choice='loss')

    rec_machine = inversefed.dgia(model,model2, (dm, ds), config, num_images=num_images)
    output, stats = rec_machine.reconstruct(input_gradient,input_gradient2, labels, img_shape=(3, 32, 32))

    test_mse = (output.detach() - ground_truth).pow(2).mean()
    feat_mse = (model(output.detach())- model(ground_truth)).pow(2).mean()  
    test_psnr = inversefed.metrics.psnr(output, ground_truth, factor=1/ds)
    print(f"Rec. loss: {stats['opt']:2.4f} | MSE: {test_mse:2.4f} "
        f"| PSNR: {test_psnr:4.2f} | FMSE: {feat_mse:2.4e} |");

    output=output.clone().detach()
    output.mul_(ds).add_(dm).clamp_(0, 1)
    output=output[0,:,:,:].permute(1, 2, 0).cpu()
    im =Image.fromarray(np.uint8(output*256))
    im.save('cf100_alex/d_{}.png'.format(node))
