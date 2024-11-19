import os
import argparse
import numpy as np
# from tqdm import tqdm
from node_parallel import node
import random
import scipy.io as scio
import random
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.autograd as autograd
from torch.utils.data import Subset,Dataset

import collections
import pickle
from model import vgg11_bn
import concurrent.futures

class args:
    batchsize = 250
    num_batch=50
    num_of_clients = 8
    val_freq = 1000
    num_comm = 10000
    IID = 1  # 1-iid  0-non-iid
    z_std=0
    epo=10

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

class TensorDataset(Dataset):
    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)

if __name__=='__main__':
    # args = parser.parse_args()
    print(torch.cuda.device_count()) 
    setup_seed(2)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    devices = [torch.device(f"cuda:{i}") for i in range(args.num_of_clients)]

    # decentralized network
    radius = np.sqrt(np.log(args.num_of_clients) / args.num_of_clients)
    np.random.seed(5)
    location = np.reshape(np.random.random(2 * args.num_of_clients), (args.num_of_clients, 2))
    A = np.zeros([args.num_of_clients, args.num_of_clients], dtype=int)
    AA = np.zeros([args.num_of_clients, args.num_of_clients], dtype=int)
    for i in range(args.num_of_clients):
        for j in range(i + 1, args.num_of_clients):
            if np.sum((location[i] - location[j]) ** 2) <= radius ** 2:
                A[i][j] = 1
                A[j][i] = 1
                AA[i][j] = 1
                AA[j][i] = -1
    P = A[:]
    tmp = A[:]
    for i in range(1, args.num_of_clients):
        tmp = np.dot(tmp, A)
        P = P + tmp
    assert np.count_nonzero(P) == args.num_of_clients ** 2

    scio.savemat("network.mat",{'A': A,'AA': AA,'loc':location})

    training_dataset = torchvision.datasets.CIFAR10(
        root="./data/cifar10",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data/cifar10",
        train=False,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    )

    
    proportion=50000//args.num_of_clients//args.batchsize//args.num_batch

    if args.IID == 1:
        subset_length = len(training_dataset) // proportion
        train_subset = Subset(training_dataset, range(subset_length))
        shuffled_indices = torch.randperm(len(training_dataset) // proportion)
        # training_inputs = training_dataset.train_data[shuffled_indices]
        training_inputs = training_dataset.data[shuffled_indices]
        training_inputs = training_inputs / 128 - 1
        training_labels = torch.Tensor(training_dataset.targets)[shuffled_indices]
        # training_labels = torch.Tensor(training_dataset.train_labels)[shuffled_indices]
        split_size = len(training_dataset) // args.num_of_clients // proportion
        split_datasets = list(
            zip(
                torch.split(torch.Tensor(training_inputs), split_size),
                torch.split(torch.Tensor(training_labels), split_size)
            )
        )
        # subset_length = len(training_dataset) // proportion
        # train_subset = Subset(training_dataset, range(subset_length))
        # shuffled_indices = torch.randperm(len(training_dataset) // proportion)
        # training_inputs = torch.stack([training_dataset[i][0].permute(1,2,0) for i in shuffled_indices])
        # training_labels = [training_dataset[i][1] for i in shuffled_indices]
        # training_labels = torch.Tensor(training_labels)
        # split_size = len(training_dataset) // (args.num_of_clients * proportion)
        # split_datasets = list(
        #     zip(
        #         torch.split(training_inputs, split_size),
        #         torch.split(training_labels, split_size)
        #     )
        # )
    else:
        subset_length = len(training_dataset) // proportion
        train_subset = Subset(training_dataset, range(subset_length))
        sorted_indices = sorted(range(subset_length), key=lambda i: train_subset[i][1])
        num_shards = 2 * args.num_of_clients
        training_inputs = training_dataset.data[sorted_indices]
        # training_inputs = training_dataset.train_data[sorted_indices]
        training_inputs = training_inputs / 128 - 1
        training_labels = torch.Tensor(training_dataset.targets)[sorted_indices]
        # training_labels = torch.Tensor(training_dataset.train_labels)[sorted_indices]
        shard_size = len(training_dataset) // num_shards // proportion
        shard_inputs = list(zip(torch.split(torch.Tensor(training_inputs), shard_size),
                                torch.split(torch.Tensor(training_labels), shard_size)))
        index = list(range(num_shards))
        random.shuffle(index)
        split_datasets = []
        for i in range(args.num_of_clients):
            tmp0 = torch.cat((shard_inputs[index[i * 2]][0], shard_inputs[index[i * 2 + 1]][0]), 0)
            tmp1 = torch.cat((shard_inputs[index[i * 2]][1], shard_inputs[index[i * 2 + 1]][1]), 0)
            split_datasets.append((tmp0, tmp1))

    # with open('dataset_ni{}_N{}.pkl'.format(args.batchsize,args.num_of_clients), 'wb') as f:
    #     pickle.dump(split_datasets, f)

    nodes=[]
    for j in range(args.num_of_clients):
        nodes.append(node(AA[j, :], torch.utils.data.DataLoader(TensorDataset(split_datasets[j][0],split_datasets[j][1]), batch_size=args.batchsize, shuffle=True),Net(),devices[j]))
    #initialize z
    dual=[[] for i in range(args.num_of_clients)]
    worker_state_dict = vgg11_bn().state_dict()
    worker_state_dict = {k: v for k, v in worker_state_dict.items() if 'running_mean' not in k and 'running_var' not in k and 'num_batches_tracked' not in k}
    weight_keys =list(worker_state_dict.keys())
    for param_tensor in weight_keys:
        for n in range(args.num_of_clients):
            dual[n].append(torch.randn(worker_state_dict[param_tensor].size())*args.z_std)
    for j in range(args.num_of_clients):
        nodes[j].init_dual(dual,devices[j])

    def train_node(node, device, epoch):
        torch.cuda.synchronize(device)
        node.model.train()
        loss, yij = node.active_update(device, epoch)
        torch.cuda.synchronize(device)
        return loss, yij

    sav=[]
    cor=[]
    cor_test=[]
    for k in range(args.num_comm):

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # future_to_node = {executor.submit(train_node, nodes[i], devices[i], args.epo): i for i in range(args.num_of_clients)}
            # y_tmp=[]
            # for future in concurrent.futures.as_completed(future_to_node):
            #     i = future_to_node[future]
            #     try:
            #         loss, yij = future.result()
            #         y_tmp.append(yij)
            #     except Exception as exc:
            #         print(f"Node {i} generated an exception: {exc}")

            # sychronous
            y_tmp=[]
            for i in range(args.num_of_clients):
                nodes[i].model.train()
                loss,yij=nodes[i].active_update(devices[i],args.epo)
                y_tmp.append(yij)

            for i in range(args.num_of_clients):
                for n,j in enumerate(np.flatnonzero(nodes[i].neighbor)):
                    nodes[j].second_update(devices[i],y_tmp[i],i,j)

            if k % args.val_freq == 0:
                loader=torch.utils.data.DataLoader(train_subset)
                for i in range(args.num_of_clients):
                    nodes[i].model.eval()
                    correct = 0
                    with torch.no_grad():
                        for data, target in loader:
                            data, target = data.to(devices[i]), target.to(devices[i])
                            output = nodes[i].model(data)
                            loss += F.cross_entropy(output, target).item() 
                            pred = output.argmax(dim=1, keepdim=True) 
                            correct += pred.eq(target.view_as(pred)).sum().item()
                    loss /= len(loader.dataset)
                    print(k+1,i,loss)
                    sav.append([k,i,loss])
                    cor.append([k,i,correct])
                # scio.savemat('loss_ni{}_N{}_z{}_i{}.mat'.format(args.batchsize,args.num_of_clients,args.z_std,args.IID), {'loss':sav,'prediction':cor})

                test_loader=torch.utils.data.DataLoader(test_dataset)
                for i in range(args.num_of_clients):
                    nodes[i].model.eval()
                    test_loss = 0
                    correct = 0
                    with torch.no_grad():
                        for data, target in test_loader:
                            data, target = data.to(devices[i]), target.to(devices[i])
                            output = nodes[i].model(data)
                            test_loss += F.cross_entropy(output, target).item() 
                            pred = output.argmax(dim=1, keepdim=True) 
                            correct += pred.eq(target.view_as(pred)).sum().item()
                    test_loss /= len(test_loader.dataset)
                    cor_test.append([k,i,correct])
                    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%),i{}\n'.format(test_loss, correct, len(test_loader.dataset),
                            100. * correct / len(test_loader.dataset),i))
                scio.savemat('loss_ni{}_N{}_z{}_i{}_imagenet_alex.mat'.format(args.batchsize,args.num_of_clients,args.z_std,args.IID), {'loss':sav,'prediction':cor,'prediction_test':cor_test})

            if k % args.val_freq == 0 or (k-1) % args.val_freq == 0:
                mod=[]
                for i in range(args.num_of_clients):
                    mod.append(nodes[i].model.state_dict())
                with open('model_ni{}_N{}_t{}_z{}_iid{}_imagenet_alex.pkl'.format(args.batchsize,args.num_of_clients,k,args.z_std,args.IID), 'wb') as f:
                    pickle.dump(mod, f)

    # test    
    for i in range(args.num_of_clients):
        nodes[i].model.eval()
        test_loss = 0
        correct = 0
        test_loader=torch.utils.data.DataLoader(test_dataset)
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(devices[i]), target.to(devices[i])
                output = nodes[i].model(data)
                test_loss += F.cross_entropy(output, target).item() 
                pred = output.argmax(dim=1, keepdim=True) 
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%),i{}\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset),i))