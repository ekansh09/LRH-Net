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

def runner(idx, model, train_dataloader,valid_dataloader,start_epoch,max_epoch,lr_scheduler,lr,criterion,optimizer,scaler,threshold,save_dir,cal_acc):
    """
    Training process
    :return:
    """

    best_acc = 0.0
    best_model_name = "dummy"
    step_start = time.time()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    history = {'train_loss': [], 'train_metric_value':[],'val_loss':[],'val_metric_value':[]}
    
     
    for epoch in range(start_epoch, max_epoch):

        
        # Update the learning rate
        if lr_scheduler is not None:
            #self.lr_scheduler.step(epoch)
            logging.info('current lr: {}'.format(lr_scheduler.get_last_lr()))
        else:
            logging.info('current lr: {}'.format(lr))
        
        
        #------------------------------------------
        #                  TRAIN
        #-------------------------------------------
        
        #Stores the predictions from last epoch
                                              
        train_loss,train_metric_value = train(criterion,optimizer,model,epoch,train_dataloader, cal_acc, threshold,scaler)
        
        labels,predictions,val_loss,val_metric_value = validation(criterion, model, epoch, valid_dataloader,threshold,cal_acc)
        
        
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
            torch.save(model, best_model_name)
            
    print("==============================  model {} gives best acc {} for fold-{} ==============================  ".format(best_model_name,best_acc,idx))
    visulaise_performance(best_labels,best_predictions,history,max_epoch,threshold,valid_dataloader.dataset.class_names,valid_dataloader.dataset.classwise_sample_count)


def setting_params(idx,opt,momentum,weight_decay,lr_scheduler,lr,steps,gamma,device):
    
    datasets = {}
    datasets['train'], datasets['val'] = ECG("../input/physionet20205folds/", str(idx)).data_preprare()
    train_dataloader = DataLoader(datasets['train'],batch_size=batch_size,shuffle = True, drop_last = True)
    valid_dataloader = DataLoader(datasets['val'],batch_size=batch_size,shuffle = True)
    
    model = resnet18()

    model.to(device)
    for name, layer in model.named_modules():
        if isinstance(layer, nn.BatchNorm1d):
            layer.float()
    
    optimizer = get_optimizer(opt,model,lr,momentum,weight_decay)
    lr_scheduler_fn = get_lr_scheduler(lr_scheduler,optimizer,steps,gamma)
    cal_acc = accuracy_measuring_fn(monitor_acc)
    threshold = np.load(weight_list[idx])['arr_0']
    scaler = amp.GradScaler()
    
    logging.basicConfig(filename='./model'+str(i)+'.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
    
    runner(idx, model,train_dataloader,valid_dataloader,start_epoch,max_epoch, lr_scheduler_fn, lr,criterion,optimizer,scaler,threshold,save_dir,cal_acc)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_classes = 24
    max_epoch = 50
    no_folds = 1

    # Load the checkpoint
    start_epoch = 0
    weight_list = ['../input/physionet-2020/magic_weight0.npz', '../input/physionet-2020/magic_weight1.npz', '../input/physionet-2020/magic_weight2.npz','../input/physionet-2020/magic_weight3.npz', '../input/physionet-2020/magic_weight4.npz', '../input/physionet-2020/magic_weight_avg.npz']
    save_dir = "./"
    opt = "adam" #optimizer
    lr = 0.0003  #the initial learning rate
    momentum = 0.9  #the momentum for sgd
    weight_decay = 1e-5 #the weight decay
    lr_scheduler = "step"        #['step', 'exp', 'stepLR', 'fix', 'cos'] the learning rate schedule
    gamma = 0.1 #learning rate scheduler parameter for step and exp
    steps = "20,40"  #the learning rate decay for step and stepLR
    monitor_acc = "ecgAcc" 
    #criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss()
    #criterion = WeightedFocalLoss()

    for i in range(no_folds):
        print("------------------------ Fold-"+str(i)+" ------------------------")
        setting_params(i,opt,momentum,weight_decay,lr_scheduler,lr,steps,gamma,device)

