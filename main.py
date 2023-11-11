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
from models import resnet18, CustomResnet
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


def runner(idx,leads,selected_leads, student_model,teacher_model1,teacher_model2,train_dataloader,valid_dataloader,start_epoch,max_epoch,lr_scheduler,lr,criterion,divergence_loss_fn,optimizer,scaler,threshold,save_dir,cal_acc,alpha,temp):
    """
    Training process
    :return:
    """

    best_acc = 0.0
    best_model_name = "dummy"
    step_start = time.time()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    history = {'train_loss': [], 'train_metric_value':[],'val_loss':[],'val_metric_value':[],'distillation_loss':[]}
    
     
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
                                              
        train_loss,train_metric_value,distillation_loss = train(criterion,divergence_loss_fn,optimizer,student_model,teacher_model1,teacher_model2,epoch,train_dataloader, cal_acc, threshold,alpha,temp,leads,selected_leads,scaler)
        
        labels,predictions,val_loss,val_metric_value = validation(criterion, student_model, epoch, valid_dataloader,threshold,leads,selected_leads,cal_acc)
        
        history['distillation_loss'].append(distillation_loss)
        history['train_loss'].append(train_loss)
        history['train_metric_value'].append(train_metric_value)
        history['val_loss'].append(val_loss)
        history['val_metric_value'].append(val_metric_value)
    
        
        if val_metric_value>best_acc:
            best_acc = val_metric_value
            best_predictions = predictions
            best_labels = labels
            best_model_name = os.path.join('./', '{}-{:.4f}-best_model-{}.pth'.format(epoch, best_acc,idx))
            print("------------Saving the model,Best Accuracy at epoch ",epoch," is:",best_acc,"----------------")
            torch.save(student_model, best_model_name)
    
    print("==============================  model {} gives best acc {} for fold-{} ==============================  ".format(best_model_name,best_acc,idx))
    visulaise_performance(best_labels,best_predictions,history,max_epoch,threshold,valid_dataloader.dataset.class_names,valid_dataloader.dataset.classwise_sample_count)

def setting_params(idx, SNOMED_Score_path, leads, selected_leads, teacher1_model_path, teacher2_model_path, opt, momentum, criterion, 
    weight_decay, lr_scheduler, lr, steps, gamma, device):
    
    datasets = {}
    datasets['train'], datasets['val'] = ECG(SNOMED_Score_path, str(idx),None,None).data_preprare()
    train_dataloader = DataLoader(datasets['train'],batch_size=batch_size,shuffle = True)
    valid_dataloader = DataLoader(datasets['val'],batch_size=batch_size,shuffle = True)
    
    inp_leads = 12 if selected_leads == None else len(selected_leads)
    student_model = CustomResnet(input_channel=inp_leads).to(device)
    
    #student_model = resnet18(in_lead = inp_leads).to(device)
    teacher_model1 = torch.load(teacher1_model_path).to(device)
    teacher_model2 = torch.load(teacher2_model_path).to(device)

    for param in teacher_model1.parameters():
        param.requires_grad = False
        
    for param in teacher_model2.parameters():
        param.requires_grad = False

    if inp_leads == 12:
        print("All 12 leads are considered")
    else:
        print("The leads considered are:",selected_leads)

    
    temp = 8
    alpha = 0.2
    divergence_loss_fn = nn.KLDivLoss(reduction = "batchmean")
    optimizer = get_optimizer(opt,student_model,lr,momentum,weight_decay)
    lr_scheduler_fn = get_lr_scheduler(lr_scheduler,optimizer,steps,gamma)
    cal_acc = accuracy_measuring_fn(monitor_acc)
    threshold = np.load(weight_list[idx])['arr_0']
    scaler = torch.cuda.amp.GradScaler()
    
    logging.basicConfig(filename='./model'+str(i)+'.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
    
    runner(idx,leads,selected_leads, student_model,teacher_model1,teacher_model2,train_dataloader,valid_dataloader,start_epoch,max_epoch, lr_scheduler_fn, lr,criterion,divergence_loss_fn,optimizer,scaler,threshold,save_dir,cal_acc,alpha,temp)


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
    max_epoch = 50
    fold_idx = '3'
    data_path = "/scratch/physionet2020_5Fold_Data/"
    SNOMED_Score_path = '/scratch/physionet2020/dx_mapping_scored.csv'
    weight_list = ['/scratch/physionet2020/magic_weight0.npz', '/scratch/physionet2020/magic_weight1.npz', '/scratch/physionet2020/magic_weight2.npz','/scratch/physionet2020/magic_weight3.npz', '/scratch/physionet2020/magic_weight4.npz', '/scratch/physionet2020/magic_weight_avg.npz']

    leads = {'I':0,'II':1,'III':2,'AVR':3,'AVL':4,'AVF':5,'V1':6,'V2':7,'V3':8,'V4':9,'V5':10,'V6':11}
    six_leads = ['I','II','III','AVR','AVL','AVF']
    four_leads = ['I','II','III','V2']
    three_leads = ['I','II','V2']
    two_leads = ['I','II']

    lead_config = six_leads


    #Saving directory
    
    save_dir = '/scratch/LRH-Net_results'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

        
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
    #criterion = WeightedFocalLoss()
    
    
    for i in range(no_folds):
        print("------------------------ Fold-"+str(i)+" ------------------------")
        setting_params(i,SNOMED_Score_path, leads,lead_config, teacher1_model_path, teacher2_model_path, 
            opt, momentum, criterion, weight_decay, lr_scheduler, lr, steps, gamma, device)
    

