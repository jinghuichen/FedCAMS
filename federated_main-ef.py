#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import math
import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, update_model_inplace, test_inference
from utils import get_model, get_dataset, average_weights, exp_details, average_parameter_delta
import utils

if __name__ == '__main__':
    start_time = time.time()

    args = args_parser()
    exp_details(args)
    
    # define paths
#     out_dir_name = args.model + '_compress_' + args.dataset + args.optimizer + '_lr' + str(args.lr) + '_locallr' + str(    args.local_lr) + '_localep' + str(args.local_ep) +'_localbs' + str(args.local_bs) + '_eps' + str(args.eps) 
    file_name = '/ef_{}_{}_{}_llr[{}]_glr[{}]_eps[{}]_le[{}]_bs[{}]_iid[{}]_mi[{}]_frac[{}]_{}.pkl'.\
                format(args.dataset, args.model, args.optimizer, 
                    args.local_lr, args.lr, args.eps, 
                    args.local_ep, args.local_bs, args.iid, args.max_init, args.frac, args.compressor)
    logger = SummaryWriter('./logs/'+file_name)
    
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(1) # limit cpu use
    print ('-- pytorch version: ', torch.__version__)
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device != 'cpu':
        torch.cuda.manual_seed(args.seed)

    if not os.path.exists(args.outfolder):
        os.mkdir(args.outfolder)

    # load dataset and user groups
    train_dataset, test_dataset, num_classes, user_groups = get_dataset(args)

    # Set the model to train and send it to device.
    global_model = get_model(args.model, args.dataset, train_dataset[0][0].shape, num_classes)
    global_model.to(device)
    global_model.train()
    
    momentum_buffer_list = []
    exp_avgs = []
    exp_avg_sqs = []
    max_exp_avg_sqs = [] 
    for i, p in enumerate(global_model.parameters()):         
        momentum_buffer_list.append(torch.zeros_like(p.data.detach().clone(), dtype=torch.float, requires_grad=False))
        exp_avgs.append(torch.zeros_like(p.data.detach().clone(), dtype=torch.float, requires_grad=False))
        exp_avg_sqs.append(torch.zeros_like(p.data.detach().clone(), dtype=torch.float, requires_grad=False))
        max_exp_avg_sqs.append(torch.zeros_like(p.data.detach().clone(), dtype=torch.float, requires_grad=False)+args.max_init) # 1e-2
    
    
    ### init error -------
    e = []
    for id in range(args.num_users):
        ei = []
        for i, p in enumerate(global_model.parameters()):         
            ei.append(torch.zeros_like(p.data.detach().clone(), dtype=torch.float, requires_grad=False))
        e.append(ei)
    D =  sum(p.numel() for p in global_model.parameters())
    print('total dimension:', D)
    print('compressor:', args.compressor)
        
    
    # Training
    train_loss_sampled, train_loss, train_accuracy = [], [], []
    test_loss, test_accuracy = [], []
    start_time = time.time()
    for epoch in tqdm(range(args.epochs)):
        ep_time = time.time()
        
        local_weights, local_params, local_losses = [], [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')
        
        
        par_before = []
        for p in global_model.parameters():  # get trainable parameters
            par_before.append(p.data.detach().clone())
        # this is to store parameters before update
        w0 = global_model.state_dict()  # get all parameters, includeing batch normalization related ones
        
        
        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            
            w, p, loss = local_model.update_weights_local(
                model=copy.deepcopy(global_model), global_round=epoch)
            

            ####### add error feedback #######
            delta = utils.sub_params(p, par_before)
            tmp = utils.add_params(e[idx], delta)
            delta_out = local_model.compressSignal(tmp, D)
            e[idx] = utils.sub_params(tmp, delta_out)
            
            local_weights.append(copy.deepcopy(w))
            local_params.append(copy.deepcopy(utils.add_params(delta_out, par_before)))
            local_losses.append(copy.deepcopy(loss))
        
           

        bn_weights = average_weights(local_weights)
        global_model.load_state_dict(bn_weights)
        
        global_delta = average_parameter_delta(local_params, par_before) # calculate compression in this function
        
        update_model_inplace(
            global_model, par_before, global_delta, args, epoch, 
            momentum_buffer_list, exp_avgs, exp_avg_sqs, max_exp_avg_sqs)
        
        # report and store loss and accuracy
        # this is local training loss on sampled users
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)
        
        

        global_model.eval()
        
        
        # Test inference after completion of training
        test_acc, test_ls = test_inference(args, global_model, test_dataset)
        test_accuracy.append(test_acc)
        test_loss.append(test_ls)

        # print global training loss after every rounds
        print('Epoch Run Time: {0:0.4f} of {1} global rounds'.format(time.time()-ep_time, epoch+1))
        print(f'Training Loss : {train_loss[-1]}')
        print(f'Test Loss : {test_loss[-1]}')
        print(f'Test Accuracy : {test_accuracy[-1]} \n')
        logger.add_scalar('train loss', train_loss[-1], epoch)
        logger.add_scalar('test loss', test_loss[-1], epoch)
        logger.add_scalar('test acc', test_accuracy[-1], epoch)
        
        if args.save:
            # Saving the objects train_loss and train_accuracy:
            

            with open(args.outfolder + file_name, 'wb') as f:
                pickle.dump([train_loss, test_loss, test_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    
