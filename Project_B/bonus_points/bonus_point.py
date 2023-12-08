# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 17:06:02 2023

@author: Jackyy
"""

######### BONUS POINT #####################
## Try IPC = 1000 and IPC = 2000

import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, \
    get_daparam, match_loss, get_time, TensorDataset, epoch, Logger
from MHIST import get_mhist_dataset
from sklearn.metrics import f1_score
import numpy as np
import sys
from torch.optim.lr_scheduler import CosineAnnealingLR
import scipy
import matplotlib.pyplot as plt


dataset = 'MNIST'  ## Choose dataset 'MNIST' or 'MHIST'
method = 'DATM'
ipc = 3000 
init_S = 'samples_predicted_correctly'  ## Initialization for Synthetic data 'real' or 'noise'
syn_train_model = 'ConvNet3'
aug = True

batch_size_syn = 32
batch_size_whole = 256 
num_epochs = 20
lr = 0.01
num_eval = 5

## Output Log
out_log = 'results/%s_%s_%s_%dipc_aug=%s_bonus_res.txt'%(dataset, init_S, method, ipc, aug) 
sys.stdout = Logger(out_log)

dist_data_path = 'results'
eval_model_pool = ['ConvNet3']

## Data Training Function (for real and distilled data)
def evaluate_model(net, trainloader_data, testloader, 
                    batch_train, lr_net, num_epochs, dataset, 
                    data_type, device = 'cuda'):
    net = net.to(device)
    if data_type == 'syn':
        images_train = trainloader_data[0].to(device)
        labels_train = trainloader_data[1].to(device)
        
        dst_train = TensorDataset(images_train, labels_train)
        trainloader = torch.utils.data.DataLoader(dst_train, batch_size=batch_train, 
                                                  shuffle=True, num_workers=0)
    elif data_type == 'real':
        trainloader = copy.deepcopy(trainloader_data)
        
    lr = float(lr_net)
    Epoch = int(num_epochs)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.5)
    criterion = nn.CrossEntropyLoss().to(device)
    lr_schedule = CosineAnnealingLR(optimizer, T_max=num_epochs)

    start = time.time()
    for ep in range(Epoch):
        loss_train, score_train = epoch('train', trainloader, net, optimizer, criterion, dataset, aug = aug)        
        lr_schedule.step()
        
        # Calculate epoch accruacy for MNIST/MHIST
        if dataset == 'MNIST':
            print(f'Epoch [{ep+1}/{num_epochs}], Loss: {loss_train:.4f}, Accuracy: {score_train:.2f}')
        elif dataset == 'MHIST':
            wf1score = score_train[-1]
            print(f'Epoch [{ep+1}/{num_epochs}], Loss: {loss_train:.4f}, F1Score (weighted): {wf1score:.2f}')

    time_train = time.time() - start
    loss_test, score_test = epoch('test', testloader, net, optimizer, criterion, dataset)
    if dataset == 'MNIST':
        print('%s Evaluate: epoch = %04d train time = %.4f s train loss = %.6f train acc = %.4f, test acc = %.4f' % (get_time(), 
                                                        Epoch, time_train, loss_train, score_train, score_test))
    else:
        print('%s Evaluate: epoch = %04d train time = %.4f s train loss = %.6f train f1 (weighted) = %.4f, test f1 (weighted) = %.4f' % (get_time(), 
                        Epoch, time_train, loss_train, score_train[-1], score_test[-1]))

    return net, score_train, score_test, loss_train, loss_test

## Get Whole and Synthesized Dataset
print('##### LOADING %s ############'%(dataset))
## Whole Dataset
channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = \
                get_dataset('MNIST', 'data')
trainloaderW = DataLoader(dst_train, batch_size=batch_size_whole, shuffle=True) 
    
## Distilled Dataset
exp_name = '%s_res_%s_%s_%s_%dipc.pt'%(method, dataset, syn_train_model, init_S, ipc)
print('[INFO] Distilled Dataset file %s'%(exp_name))
dist_data = torch.load(os.path.join(dist_data_path, exp_name))['data']
    
    
############### LOOP OVER NETWORKS AND TRAIN ON BOTH DATASET ##################
loss_whole_data, score_whole_data = [], []
loss_dist_data, score_dist_data = [], []
for model in eval_model_pool:  
    running_loss_whole, running_score_whole = 0, 0
    running_loss_dist, running_score_dist = 0, 0
    for it in range(num_eval):
        ## Evaluate Model on Synthetic dataset
        print("[INFO] Iteration %d: Training Model %s on Distilled dataset"%(it+1, model))
        net = get_network(model, channel, num_classes, im_size=im_size)  
        _, score_train, score_test, loss_train, loss_test = evaluate_model(net, 
             dist_data, testloader, batch_size_syn, lr, num_epochs, dataset, 'syn')
        running_loss_dist += loss_test
        
        if dataset == 'MNIST':
            running_score_dist += score_test
        elif dataset == 'MHIST':
            running_score_dist += score_test[-1]  ##  Use weighted F1 Score
        
        print("[INFO] Iteration %d: Training Model %s on whole dataset"%(it+1, model))
        ## Evaluate Model on Whole dataset
        net = get_network(model, channel, num_classes, im_size=im_size)  
        _, score_train, score_test, loss_train, loss_test = evaluate_model(net, 
             trainloaderW, testloader, batch_size_syn, lr, num_epochs, dataset, 'real')
        running_loss_whole += loss_test
        
        if dataset == 'MNIST':
            running_score_whole += score_test
        elif dataset == 'MHIST':
            running_score_whole += score_test[-1]
        
    
    print('------------------------------------------------------------------')
    print('Average Test Score on Distilled Dataset is %.4f'%(running_score_dist/num_eval))
    print('Average Test Score on Whole Dataset is %.4f'%(running_score_whole/num_eval))
    print('------------------------------------------------------------------')
    
    loss_whole_data.append(running_loss_whole/num_eval)
    score_whole_data.append(running_score_whole/num_eval)
    
    loss_dist_data.append(running_loss_dist/num_eval)
    score_dist_data.append(running_score_dist/num_eval)


    
sys.stdout.close()



