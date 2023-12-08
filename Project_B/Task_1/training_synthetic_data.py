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
    model = 'ConvNet3' if dataset == 'MNIST' else 'ConvNet7'
    ipc = 10 if dataset == 'MNIST' else 50  ## Number of synthetic images per class
    K = 100 if dataset == 'MNIST' else 200  ## Nuber of random weight initializations
    batch_size = 256 if dataset == 'MNIST' else 128 ## minibatch size
    T = 10   ## Number of iterations
    lr_S = 0.1 ## Synthetic data learning rate
    opt_steps_S = 1  ## Number of optimizer steps for Synthetic data
    lr_M = 0.01  ## Model learning rate
    opt_steps_M = 50 ## Number of optimizer steps for Model
    device = 'cuda'  ## Select Computing device 'cuda' or 'cpu'
    init_S = 'real'  ## Initialization for Synthetic data 'real' or 'noise'
    save_path = 'results'  ## Str or None
    
    
    out_log = 'results/%s_%s_Synthetic_dataset.txt'%(dataset, init_S) 
    sys.stdout = Logger(out_log)
    
    if dataset == 'MNIST':
        model_eval_pool = [model, 'VGG11']
    elif dataset == 'MHIST':
        model_eval_pool = [model, 'VGG11_mhist']
    
    
    ## Training Parameters for Evaluation on Synthetic Data
    batch_train = 32  ### Evaluation Batch size for Synthetic data
    lr_net = 0.01
    epoch_eval_train = 20
    
    ## Numbers
    Iteration = K
    outer_loop = T
    inner_loop = opt_steps_M

    # if not os.path.exists(args.data_path):
    #     os.mkdir(args.data_path)

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    if dataset == 'MNIST':
        eval_it_pool = [0, Iteration//2, Iteration] 
    elif dataset == 'MHIST':
        eval_it_pool = np.arange(0, Iteration+1, Iteration//40).tolist()
    save_image_pool = [0, Iteration//2, Iteration] 
    
    print('eval_it_pool: ', eval_it_pool)
    acc_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        acc_all_exps[key] = []

    if dataset == 'MNIST':
        channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset('MNIST', 'data')
    elif dataset =='MHIST':
        data_path = 'C:\\Users\Jackyy\OneDrive - University of Toronto\D\DIP\projectB_sh\DatasetCondensation-modified\mhist_dataset'
        ann_file = 'annotations.csv'
        channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test = get_mhist_dataset(data_path, ann_file)
        testloader = DataLoader(dst_test, batch_size=batch_size, shuffle=False)       

    ''' organize the real dataset '''
    images_all = []
    labels_all = []
    indices_class = [[] for c in range(num_classes)]

    images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
    labels_all = [dst_train[i][1] for i in range(len(dst_train))]
    for i, lab in enumerate(labels_all):
        indices_class[lab].append(i)
    images_all = torch.cat(images_all, dim=0).to(device)
    labels_all = torch.tensor(labels_all, dtype=torch.long, device=device)
    
    # ## Display Mean and STD of training data
    # mean = torch.mean(images_all, dim = (0, 2, 3)).to('cpu').numpy()
    # std = torch.std(images_all, dim = (0, 2, 3)).to('cpu').numpy()
    # print('Train Data Mean and STD ....')
    # print(mean)
    # print(std)

    for c in range(num_classes):
        print('class c = %d: %d real images'%(c, len(indices_class[c])))

    def get_images(c, n): # get random n images from class c
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        return images_all[idx_shuffle]

    for ch in range(channel):
        print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))


    ''' initialize the synthetic data '''
    image_syn = torch.randn(size=(num_classes*ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=device)
    label_syn = torch.tensor([np.ones(ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=device).view(-1) 

    if init_S == 'real':
        print('initialize synthetic data from random real images')
        for c in range(num_classes):
            image_syn.data[c*ipc:(c+1)*ipc] = get_images(c, ipc).detach().data
    else:
        print('initialize synthetic data from random noise')


    ''' training '''
    optimizer_img = torch.optim.SGD([image_syn, ], lr=lr_S, momentum=0.5) # optimizer_img for synthetic data
    optimizer_img.zero_grad()
    criterion = nn.CrossEntropyLoss().to(device)
    print('%s training begins'%get_time())

    for it in range(Iteration+1):
        
        ''' Evaluate synthetic data '''
        if it in eval_it_pool:
            for model_eval in model_eval_pool:
                print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(model, model_eval, it))                
                
                net_eval = get_network(model_eval, channel, num_classes, im_size).to(device) # get a random model
                image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach()) # avoid any unaware modification
                _, score_train, score_test = evaluate_synset(net_eval, image_syn_eval, label_syn_eval, testloader,
                                                         batch_train, lr_net, epoch_eval_train, dataset)
                #print('Evaluate on Model: %s, train score = %.4f test score = %.4f\n-----------------'%(model_eval, acc_train, acc_test))
                print('----------------------------------------')
                acc_all_exps[model_eval].append(score_test)
        
        if it in save_image_pool:
            if save_path != None:
                ''' visualize and save '''
                save_name = os.path.join(save_path, 'vis_%s_%s_%s_%dipc_iter%d.png'%(dataset, model, init_S, ipc, it))
                image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
                for ch in range(channel):
                    image_syn_vis[:, ch] = image_syn_vis[:, ch]  * std[ch] + mean[ch]
                image_syn_vis[image_syn_vis<0] = 0.0
                image_syn_vis[image_syn_vis>1] = 1.0
                save_image(image_syn_vis, save_name, nrow=ipc) # Trying normalize = True/False may get better visual effects.
    
        ''' Train synthetic data '''
        net = get_network(model, channel, num_classes, im_size).to(device) # get a random model
        net.train()
        net_parameters = list(net.parameters())
        optimizer_net = torch.optim.SGD(net.parameters(), lr=lr_M)  # optimizer_img for synthetic data
        optimizer_net.zero_grad()
        loss_avg = 0

        for ol in range(outer_loop):

            ''' freeze the running mu and sigma for BatchNorm layers '''
            # Synthetic data batch, e.g. only 1 image/batch, is too small to obtain stable mu and sigma.
            # So, we calculate and freeze mu and sigma for BatchNorm layer with real data batch ahead.
            # This would make the training with BatchNorm layers easier.

            BN_flag = False
            BNSizePC = 16  # for batch normalization
            for module in net.modules():
                if 'BatchNorm' in module._get_name(): #BatchNorm
                    BN_flag = True
            if BN_flag:
                img_real = torch.cat([get_images(c, BNSizePC) for c in range(num_classes)], dim=0)
                net.train() # for updating the mu, sigma of BatchNorm
                output_real = net(img_real) # get running mu, sigma
                for module in net.modules():
                    if 'BatchNorm' in module._get_name():  #BatchNorm
                        module.eval() # fix mu and sigma of every BatchNorm layer

            ''' update synthetic data '''
            loss = torch.tensor(0.0).to(device)
            for c in range(num_classes):
                img_real = get_images(c, batch_size)
                lab_real = torch.ones((img_real.shape[0],), device=device, dtype=torch.long) * c
                img_syn = image_syn[c*ipc:(c+1)*ipc].reshape((ipc, channel, im_size[0], im_size[1]))
                lab_syn = torch.ones((ipc,), device=device, dtype=torch.long) * c

                output_real = net(img_real)
                loss_real = criterion(output_real, lab_real)
                gw_real = torch.autograd.grad(loss_real, net_parameters)
                gw_real = list((_.detach().clone() for _ in gw_real))

                output_syn = net(img_syn)
                loss_syn = criterion(output_syn, lab_syn)
                gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)

                loss += match_loss(gw_syn, gw_real, dis_metric = 'ours')

            optimizer_img.zero_grad()
            loss.backward()
            optimizer_img.step()
            loss_avg += loss.item()

            if ol == outer_loop - 1:
                break

            ''' update network '''
            image_syn_train, label_syn_train = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach())  # avoid any unaware modification
            dst_syn_train = TensorDataset(image_syn_train, label_syn_train)
            trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=batch_size, shuffle=True)
            for il in range(inner_loop):
                epoch('train', trainloader, net, optimizer_net, criterion, dataset)


        loss_avg /= (num_classes*outer_loop)

        if it%1 == 0:
            print('%s iter = %04d, loss = %.4f' % (get_time(), it, loss_avg))

        if it == Iteration: # only record the final results
            if save_path != None:
                data_save = copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())
                torch.save({'data': data_save, 'acc_all_exps': acc_all_exps}, os.path.join(save_path, 'GM_%s_%s_%s_%dipc.pt'%(dataset, model, init_S, ipc)))


    sys.stdout.close()



if __name__ == '__main__':
    main()


