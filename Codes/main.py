import datasets
from torch.utils.data import Dataset, DataLoader
import torch

import logging
import warnings
import time
import numpy as np
import os

from sklearn.metrics import roc_auc_score, f1_score, cohen_kappa_score 
from torchsummary import summary
from functools import partial
import torch.nn.functional as F


from Loading_dataset import ECG
from models import resnet18
from utils import get_optimizer, accuracy_measuring_fn, get_lr_scheduler
from Evaluate_and_Visualise_performance import visulaise_performance


def get_dataset(batch_size, fold_index, data_path, data_score_path):
    
    datasets = {}
    datasets['train'], datasets['val'] = ECG(data_path,data_score_path, fold_index).data_preprare()

    print('Validation length:',len(datasets['val']), ' Training Length:',len(datasets['train']))

    train_dataloader = DataLoader(datasets['train'],batch_size= batch_size,shuffle = True)
    valid_dataloader = DataLoader(datasets['val'],batch_size=batch_size,shuffle = True)

    signals, class_labels,age_gender,indices = next(iter(train_dataloader))
    
    print('\n\n-------- Sample data ---------')
    print('Singnals sample length:', signals.shape)
    print('class labels:', class_labels.shape)
    print('age and gender shape',age_gender.shape, '\n\n')    
    
    return train_dataloader, valid_dataloader


def runner(idx, model, train_dataloader,valid_dataloader,start_epoch,max_epoch,lr_scheduler,lr,criterion,optimizer,threshold,save_dir):
    """
    Training process
    :return:
    """

    best_acc = 0.0
    best_model_name = "dummy"
    history = {'train_loss': [], 'train_auprc':[],'val_loss':[],'val_auprc':[]}
    
     
    for epoch in range(start_epoch, max_epoch):

        
        #Update the learning rate
        if lr_scheduler is not None:
            lr_scheduler.step(epoch)
            logging.info('current lr: {}'.format(lr_scheduler.get_last_lr()))
        else:
            logging.info('current lr: {}'.format(lr))
        
        
        #------------------------------------------
        #                  TRAIN
        #-------------------------------------------
        
        #Stores the predictions from last epoch
                                              
        train_loss,train_auprc = train(criterion,optimizer,model,epoch,train_dataloader)
        
        labels,predictions,val_loss,val_auprc = validation(criterion, model, epoch, valid_dataloader,batch_size)
        
        
        history['train_loss'].append(train_loss)
        history['train_auprc'].append(train_auprc)
        history['val_loss'].append(val_loss)
        history['val_auprc'].append(val_auprc)
    
        
        if val_auprc>best_acc:
            best_acc = val_auprc
            best_predictions = predictions
            best_labels = labels
            best_model_name = os.path.join('./', '{}-{:.4f}-best_model-{}.pth'.format(epoch, best_acc,idx))
            print("------------Saving the model,Best auprc at epoch ",epoch," is:",best_acc,"----------------")
            torch.save(model, best_model_name)
    
    print('Loading Thresholds...')
    threshold = find_thresholds(best_labels.cpu().detach().numpy(), best_predictions.cpu().detach().numpy(), valid_dataloader.dataset.class_codes.tolist())
    print('Thresholds generated...')
    
    with open(save_dir+'/threshold.txt', 'w') as f:                                                                                                
        f.write(' '.join(map(str,threshold))) 

    
    print("==============================  model {} gives best acc {} for fold-{} ==============================  ".format(best_model_name,best_acc,idx))
    visulaise_performance(best_labels,best_predictions,history,max_epoch,threshold,valid_dataloader.dataset.class_names,valid_dataloader.dataset.classwise_sample_count)


if __name__ == "__main__":


    batch_size = 64

    #Loading the GPU or CPU 
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_count = torch.cuda.device_count()
        logging.info('using {} gpus'.format(device_count))
        assert batch_size % device_count == 0, "batch size should be divided by device count"
    else:
        warnings.warn("gpu is not available")
        device = torch.device("cpu")
        device_count = 1
        logging.info('using {} cpu'.format(device_count))

    print('The device loaded is:',device,'  whose count is"',device_count)
    
    
    #Loading dataset
    batch_size = 64
    start_epoch = 0
    max_epoch = 2
    fold_idx = '3'
    data_path = "/scratch/physionet2020_5Fold_Data/"
    SNOMED_Score_path = '/scratch/physionet2020/dx_mapping_scored.csv'
    weight_list = ['/scratch/physionet2020/magic_weight0.npz', '/scratch/physionet2020/magic_weight1.npz', '/scratch/physionet2020/magic_weight2.npz','/scratch/physionet2020/magic_weight3.npz', '/scratch/physionet2020/magic_weight4.npz', '/scratch/physionet2020/magic_weight_avg.npz']
    train_loader, valid_loader =  get_dataset(batch_size, fold_idx, data_path, SNOMED_Score_path)
    threshold = np.load(weight_list[int(fold_idx)])['arr_0']

    #Saving directory
    
    save_dir = '/scratch/LRH-Net_results'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    
    #Loading the resnet model
    model = resnet18()
    model.to(device)
    
    #Loading the model configuration
    opt = "adam" #optimizer
    lr = 0.003  #the initial learning rate
    momentum = 0.9  #the momentum for sgd
    weight_decay = 1e-5 #the weight decay
    lr_scheduler = "step"        #['step', 'exp', 'stepLR', 'fix', 'cos'] the learning rate schedule
    gamma = 0.1 #learning rate scheduler parameter for step and exp
    steps = "20,40,70"  #the learning rate decay for step and stepLR
    monitor_acc = "ecgAcc" 
    criterion = nn.BCEWithLogitsLoss()

    optimizer = get_optimizer(opt,model,lr,momentum,weight_decay)
    
    #lr_scheduler_fn = get_lr_scheduler(lr_scheduler,optimizer,steps,gamma)
    lr_scheduler_fn = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=max_epoch)
    
    cal_acc = accuracy_measuring_fn(monitor_acc)
    
    runner(fold_idx, model,train_loader,valid_loader,start_epoch,max_epoch, lr_scheduler_fn, lr,criterion,optimizer,threshold,save_dir)

    
    

