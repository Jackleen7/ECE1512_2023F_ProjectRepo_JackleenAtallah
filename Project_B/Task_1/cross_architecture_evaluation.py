# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 19:36:53 2023

@author: Jackyy
"""

##### Cross-Architecture Evaluation ##########
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

def main():
    
    ## Training Parameters For Synthetic Data
    dataset = 'MNIST'  ## Choose dataset 'MNIST' or 'MHIST'
    method = 'DATM' # GM for Gradient matching or DATM for difficulty alignment trajectory matching
    ipc = 10 if dataset == 'MNIST' else 50  ## Number of synthetic images per class
    device = 'cuda'  ## Select Computing device 'cuda' or 'cpu'
    init_S = 'real'  ## Initialization for Synthetic data 'real' or 'noise'
    save_path = 'results'  ## Str or None    
    num_eval = 5
    
    out_log = 'results/%s_%s_%s_cross_architecture_res.txt'%(dataset, init_S, method) 
    sys.stdout = Logger(out_log)
    
    if dataset == 'MNIST':
        model_eval_pool = ['ConvNet3', 'VGG11', 'AlexNet', 'ResNet18', 'LeNet']
        model_train = 'ConvNet3'
    elif dataset == 'MHIST':
        model_eval_pool = ['ConvNet7', 'VGG11_mhist']
        model_train = 'ConvNet7'
    syn_data_file = '%s_res_%s_%s_%s_%dipc.pt'%(method, dataset, model_train, init_S, ipc)
    print("Synthetic Data File: %s"%(syn_data_file))
    
    ## Training Parameters for Evaluation on Synthetic Data
    batch_size = 32  ### Evaluation Batch size for Synthetic data
    lr_net = 0.01
    epoch_eval_train = 100
    
    
    if dataset == 'MNIST':
        channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset('MNIST', 'data')
    elif dataset =='MHIST':
        data_path = 'C:\\Users\Jackyy\OneDrive - University of Toronto\D\DIP\projectB_sh\DatasetCondensation-modified\mhist_dataset'
        ann_file = 'annotations.csv'
        channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test = get_mhist_dataset(data_path, ann_file)
        testloader = DataLoader(dst_test, batch_size=batch_size, shuffle=False)       
    
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        
    score_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        score_all_exps[key] = []

    ########### LOAD SYNTHETIC DATA #############################
    syn_data = torch.load(os.path.join("results", syn_data_file))["data"]
    image_syn, label_syn = syn_data[0], syn_data[1]

    for model_eval in model_eval_pool:
        
        if dataset == 'MNIST':
            avg_test_acc = 0
        elif dataset == 'MHIST':
            avg_ssa_f1, avg_hp_f1, avg_wf1 = 0, 0, 0
        
        for it in range(num_eval):
            print('----Evaluation model_train = %s, model_eval = %s, iteration = %d'%(model_train, model_eval, it+1))                
            net_eval = get_network(model_eval, channel, num_classes, im_size).to(device) # get a random model
            image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach()) # avoid any unaware modification
            _, score_train, score_test = evaluate_synset(net_eval, image_syn_eval, label_syn_eval, testloader,
                                                 batch_size, lr_net, epoch_eval_train, dataset)
            
            if dataset == 'MNIST':
                avg_test_acc += score_test
            elif dataset == 'MHIST':
                avg_hp_f1 += score_test[0] 
                avg_ssa_f1 += score_test[1] 
                avg_wf1 += score_test[2]
                
        print('---------------- Average Results  -------------')
        if dataset == 'MNIST':
            print('Average Test accuracy is %.4f'%(avg_test_acc/num_eval))
        elif dataset == 'MHIST':
            print('Average SSA_F1: %.4f, Average HP_F1: %.4f, Average wF1: %.4f'%(avg_ssa_f1/num_eval,
                         avg_hp_f1/num_eval, avg_wf1/num_eval))
        print('--------------------------------------------------------------')
        print('--------------------------------------------------------------')

    sys.stdout.close()



if __name__ == '__main__':
    main()


