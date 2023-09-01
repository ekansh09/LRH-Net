import torch
import torch as T
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
import numpy as np
import pandas as pd
import os
from scipy.signal import resample
from scipy.stats import zscore
from sklearn.metrics import average_precision_score,precision_recall_curve,roc_curve
import sklearn.metrics as skm
import random
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.optimize import differential_evolution
import copy
from scipy import signal

def Resample(input_signal, src_fs, tar_fs):
    
    if src_fs != tar_fs:
        dtype = input_signal.dtype
        audio_len = input_signal.shape[1]
        audio_time_max = 1.0 * (audio_len) / src_fs
        src_time = 1.0 * np.linspace(0, audio_len, audio_len) / src_fs
        tar_time = 1.0 * np.linspace(0, np.int(audio_time_max * tar_fs), np.int(audio_time_max * tar_fs)) / tar_fs
        for i in range(input_signal.shape[0]):
            if i == 0:
                output_signal = np.interp(tar_time, src_time, input_signal[i, :]).astype(dtype)
                output_signal = output_signal.reshape(1, len(output_signal))
            else:
                tmp = np.interp(tar_time, src_time, input_signal[i, :]).astype(dtype)
                tmp = tmp.reshape(1, len(tmp))
                output_signal = np.vstack((output_signal, tmp))
    else:
        output_signal = input_signal
    return output_signal


def load_data(case, src_fs, tar_fs=257):
    #case = case.replace('..', '.')

    x = loadmat(case)
    data = np.asarray(x['val'], dtype=np.float64)
    data = Resample(data, src_fs, tar_fs)
    return data

def prepare_data(age, gender): 
    data = np.zeros(5,) #age, age_mask,female,male ,gender_mask 
    if age >= 0:
        data[0] = age / 100
        data[1] = 1
    if 'F' in gender:
        data[2] = 1
        data[4] = 1
    elif gender == 'Unknown':
        data[4] = 0
    elif 'f' in gender:
        data[2] = 1
        data[4] = 1
    else:
        data[3] = 1
        data[4] = 1

    return data

class dataset(Dataset):

    def __init__(self, anno_pd, test=False, transform=None,class_codes=None, class_names=None, data_dir=None, loader=load_data):
        self.test = test
        if self.test:
            self.data = anno_pd['filename'].tolist()
            self.fs = anno_pd['fs'].tolist()
        else:
            self.data = anno_pd['filename'].tolist()
            print('DATA:', data_dir)
            self.class_codes = class_codes
            self.class_names = class_names
            self.classwise_sample_count = [int(anno_pd.iloc[:, i].values.sum()) for i in range(4,anno_pd.shape[1])]
            self.labels = anno_pd.iloc[:, 4:].values
            self.multi_labels = [self.labels[i, :] for i in range(self.labels.shape[0])]
            self.age = anno_pd['age'].tolist()
            self.gender = anno_pd['gender'].tolist()
            self.fs = anno_pd['fs'].tolist()

            self.fs = anno_pd['fs'].tolist()

        self.transforms = transform
        self.data_dir = data_dir
        self.loader = loader


    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        if self.test:
            img_path = self.data[item]
            fs = self.fs[item]
            img = self.loader(self.data_dir + img_path, src_fs=fs)
            img = self.transforms(img)
            return img, img_path
        else:
            img_name = self.data[item]
            fs = self.fs[item]
            age = self.age[item]
            gender = self.gender[item]
            age_gender = prepare_data(age, gender)
            img = self.loader(img_name, src_fs=fs)
            label = self.multi_labels[item]
        
            """
            for i in range(img.shape[1]):
                img[:, i] = ecg_preprocessing(img[:, i], wfun='db6', levels=9, type=2)
            """
            img = self.transforms(img)
            return img, torch.from_numpy(label).float(),torch.from_numpy(age_gender).float(),item

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, seq):
        for t in self.transforms:
            seq = t(seq)
        return seq
    
class Filtering(object):
    def __call__(self, seq):
        b,a = signal.butter(3, [0.001 / 250, 47 / 250], 'bandpass')
        seq = signal.filtfilt(b, a, seq)
        
        return seq
    
class ZScore(object):
    def __call__(self, seq):
        seq = zscore(seq, axis=-1)
        return seq
    
class NaNvalues(object):
    def __call__(self, seq):
        seq = np.nan_to_num(seq)
        return seq
    

class ValClip(object):
    def __init__(self, len=72000):
        self.len = len

    def __call__(self, seq):
        if seq.shape[1] >= self.len:
            seq = seq
        else:
            zeros_padding = np.zeros(shape=(seq.shape[0], self.len - seq.shape[1]), dtype=np.float32)
            seq = np.hstack((seq, zeros_padding))
        return seq
class RandomClip(object):
    def __init__(self, len=72000):
        self.len = len

    def __call__(self, seq):
        if seq.shape[1] >= self.len:
            start = random.randint(0, seq.shape[1] - self.len)
            seq = seq[:, start:start+self.len]
        else:
            left = random.randint(0, self.len - seq.shape[1])
            right = self.len - seq.shape[1] - left
            zeros_padding1 = np.zeros(shape=(seq.shape[0], left), dtype=np.float32)
            zeros_padding2 = np.zeros(shape=(seq.shape[0], right), dtype=np.float32)
            seq = np.hstack((zeros_padding1, seq, zeros_padding2))
        return seq
class Normalize(object):
    def __init__(self, type="0-1"):
        self.type = type

    def __call__(self, seq):
        if self.type == "0-1":
            for i in range(seq.shape[0]):
                if np.sum(seq[i, :]) == 0:
                    seq[i, :] = seq[i, :]
                else:
                    seq[i, :] = (seq[i, :]-seq[i, :].min())/(seq[i, :].max()-seq[i, :].min())
        elif self.type == "mean-std":
            for i in range(seq.shape[0]):
                if np.sum(seq[i, :]) == 0:
                    seq[i, :] = seq[i, :]
                else:
                    seq[i, :] = (seq[i, :]-seq[i, :].mean())/seq[i, :].std()
        elif self.type == "none":
            seq = seq
        else:
            raise NameError('This normalization is not included!')
        return seq
    
    
class Retype(object):
    def __call__(self, seq):
        return seq.astype(np.float32)

normlizetype = 'mean-std'
start = 0
seq_length = 4096
sample_ratio = 0.5
data_transforms = {
    'train': Compose([
        #Reshape(),
        #DownSample(sample_ratio),
        #ZerosPadding(len=seq_length),
        # ConstantStart(start=start, num=seq_length),
        Filtering(),
        ZScore(),  #gives NAN vales
        NaNvalues(),
        Normalize(normlizetype),
        RandomClip(len=seq_length),
        # RandomAddGaussian(),
        # RandomScale(0.1),
        # RandomAmplify(),
        # Randomverflip(),
        # Randomshift(),
        # RandomStretch(0.02),
        # RandomCrop(),
        Retype()
    ]),
    'val': Compose([
        #Reshape(),
        #DownSample(sample_ratio),
        #RandomClip(len=seq_length),
        Filtering(),
        ZScore(),  #gives NAN vales
        NaNvalues(),
        Normalize(normlizetype),
        RandomClip(len=seq_length),
        #ValClip(len=seq_length),           **********(CHECK)
        #ZerosPadding(len=seq_length),
        # ConstantStart(start=start, num=seq_length),
        
        Retype()
    ]),
    'test': Compose([
        #Reshape(),
        #DownSample(sig_resample_len),
        Filtering(),
        ZScore(),  #gives NAN vales
        NaNvalues(),
        Normalize(normlizetype),
        RandomClip(len=seq_length),
        #ValClip(len=seq_length),
        Retype()
    ])
}

def load_and_clean_sub(path):
    label_test = pd.read_csv(path, sep="\t", names=list(range(0, 3)))
    label_test.columns = ['id', 'age', 'gender']  
    return label_test

class ECG(object):
    


    def __init__(self, data_dir, split='0'):
        self.data_dir = data_dir
        self.split = split
        self.num_classes = 24
        self.inputchannel = 12


    def data_preprare(self, test=False):
        if test:
            train_path = self.data_dir+'train_split' + self.split + '.csv'
            val_path = self.data_dir+'test_split' + self.split + '.csv'
            train_pd = pd.read_csv(train_path)
            val_pd = pd.read_csv(val_path)

            train_dataset = dataset(anno_pd=train_pd, transform=data_transforms['train'], data_dir=self.data_dir)
            val_dataset = dataset(anno_pd=val_pd, transform=data_transforms['val'], data_dir=self.data_dir)
            return train_dataset, val_dataset
        else:
            train_path = self.data_dir+'train_split' + self.split + '.csv'
            val_path = self.data_dir+'test_split' + self.split + '.csv'
            train_pd = pd.read_csv(train_path)
            val_pd = pd.read_csv(val_path)
            codes_mapping = pd.read_csv('/scratch/physionet2020/dx_mapping_scored.csv')
            class_codes = train_pd.columns.values[4:]
            class_names = [codes_mapping[codes_mapping['SNOMED CT Code']==int(label)]['Dx'].values[0] for label in class_codes]
            #print('ECG:', self.data_dir)
            
            
            train_dataset = dataset(anno_pd=train_pd, transform=data_transforms['train'],class_codes=class_codes,class_names=class_names, data_dir=self.data_dir)
            val_dataset = dataset(anno_pd=val_pd, transform=data_transforms['val'],class_codes=class_codes,class_names = class_names, data_dir=self.data_dir)
            return train_dataset, val_dataset

import datasets

datasets = {}
batch_size = 64
datasets['train'], datasets['val'] = ECG("/scratch/physionet2020_5Fold_Data/", "3").data_preprare()

print(len(datasets['val']))

train_dataloader = DataLoader(datasets['train'],batch_size= batch_size,shuffle = True)
valid_dataloader = DataLoader(datasets['val'],batch_size=batch_size,shuffle = True)

signals, class_labels,age_gender,indices = next(iter(train_dataloader))

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import torch.optim as optim
import torch
from torch.optim import lr_scheduler
import logging
import warnings
import time

from sklearn.metrics import roc_auc_score, f1_score, cohen_kappa_score 
from torchsummary import summary
from functools import partial
import torch.nn.functional as F

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

def conv3x1(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=3, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,is_last = False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = conv3x1(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x1(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = SELayer(planes)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(.2)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        preact = out
        out = self.relu(out)

        if self.is_last:
            return out, preact
        else:
            return out

class ResNet(nn.Module):

    def __init__(self, block, layers, in_channel=12, out_channel=24, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv1d(in_channel, 64, kernel_size=15, stride=2, padding=7,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(5, 10)
        self.fc = nn.Linear(512 * block.expansion + 10, out_channel)
        self.sig = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        """
                          
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        
        layers.append(block(self.inplanes, planes, stride, downsample,is_last=(blocks == 1)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            
            layers.append(block(self.inplanes, planes,is_last=(i == blocks-1)))

        return nn.Sequential(*layers)



    def forward(self, x, ag, is_feat=False, preact=False):
        
        #print("input:",x.shape, ag.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        f0 = x
        x = self.maxpool(x)
        f1 = x

        x, f2_pre = self.layer1(x)
        f2 = x
        x, f3_pre = self.layer2(x)
        f3 = x
        x, f4_pre = self.layer3(x)
        f4 = x
        x, f5_pre = self.layer4(x)
        f5 = x

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        f6 = x
        
        ag = self.fc1(ag)
        x = torch.cat((ag, x), dim=1)
        #print("x:",x.shape)
        x = self.fc(x)
        #x = self.sig(x)

        if is_feat:
            if preact:
                return [f0, f1, f2_pre, f3_pre, f4_pre, f5_pre, f6], x
            else:
                return [f0, f1, f2, f3, f4,f5,f6], x
        else:
            return x

def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

# Check if the input is a number.
def is_number(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


# Load a table with row and column names.
def load_table(table_file):
    # The table should have the following form:
    #
    # ,    a,   b,   c
    # a, 1.2, 2.3, 3.4
    # b, 4.5, 5.6, 6.7
    # c, 7.8, 8.9, 9.0
    #
    table = list()
    with open(table_file, 'r') as f:
        for i, l in enumerate(f):
            arrs = [arr.strip() for arr in l.split(',')]
            table.append(arrs)

    # Define the numbers of rows and columns and check for errors.
    num_rows = len(table)-1
    if num_rows<1:
        raise Exception('The table {} is empty.'.format(table_file))

    num_cols = set(len(table[i])-1 for i in range(num_rows))
    if len(num_cols)!=1:
        raise Exception('The table {} has rows with different lengths.'.format(table_file))
    num_cols = min(num_cols)
    if num_cols<1:
        raise Exception('The table {} is empty.'.format(table_file))

    # Find the row and column labels.
    rows = [table[0][j+1] for j in range(num_rows)]
    cols = [table[i+1][0] for i in range(num_cols)]

    # Find the entries of the table.
    values = np.zeros((num_rows, num_cols))
    for i in range(num_rows):
        for j in range(num_cols):
            value = table[i+1][j+1]
            if is_number(value):
                values[i, j] = float(value)
            else:
                values[i, j] = float('nan')

    return rows, cols, values



# Load weights.
def load_weights(weight_file, classes):
    # Load the weight matrix.
    rows, cols, values = load_table(weight_file)
    assert(rows == cols)
    num_rows = len(rows)

    # Assign the entries of the weight matrix with rows and columns corresponding to the classes.
    num_classes = len(classes)
    weights = np.zeros((num_classes, num_classes), dtype=np.float64)
    for i, a in enumerate(rows):
        if a in classes:
            k = classes.index(a)
            for j, b in enumerate(rows):
                if b in classes:
                    l = classes.index(b)
                    weights[k, l] = values[i, j]

    return weights

# Compute modified confusion matrix for multi-class, multi-label tasks.
def compute_modified_confusion_matrix(labels, outputs):
    # Compute a binary multi-class, multi-label confusion matrix, where the rows
    # are the labels and the columns are the outputs.
    num_recordings, num_classes = np.shape(labels)
    A = np.zeros((num_classes, num_classes))

    # Iterate over all of the recordings.
    for i in range(num_recordings):
        # Calculate the number of positive labels and/or outputs.
        normalization = float(max(np.sum(np.any((labels[i, :], outputs[i, :]), axis=0)), 1))
        # Iterate over all of the classes.
        for j in range(num_classes):
            # Assign full and/or partial credit for each positive class.
            if labels[i, j]:
                for k in range(num_classes):
                    if outputs[i, k]:
                        A[j, k] += 1.0/normalization

    return A


# Compute the evaluation metric for the Challenge.
def compute_challenge_metric(weights, labels, outputs, classes, normal_class):
    num_recordings, num_classes = np.shape(labels)
    normal_index = classes.index(normal_class)

    # Compute the observed score.
    A = compute_modified_confusion_matrix(labels, outputs)
    observed_score = np.nansum(weights * A)

    # Compute the score for the model that always chooses the correct label(s).
    correct_outputs = labels
    A = compute_modified_confusion_matrix(labels, correct_outputs)
    correct_score = np.nansum(weights * A)

    # Compute the score for the model that always chooses the normal class.
    inactive_outputs = np.zeros((num_recordings, num_classes), dtype=np.bool)
    inactive_outputs[:, normal_index] = 1
    A = compute_modified_confusion_matrix(labels, inactive_outputs)
    inactive_score = np.nansum(weights * A)

    if correct_score != inactive_score:
        normalized_score = float(observed_score - inactive_score) / float(correct_score - inactive_score)
    else:
        normalized_score = float('nan')

    return normalized_score

#F1score
def cal_Acc(y_true, y_pre, threshold=0.5, num_classes=9, beta=2, normal=False):
    
    y_true = y_true.cpu().detach().numpy().astype(np.int)

    y_label = np.zeros(y_true.shape)
    # Generate the one hot encoding labels
    _, y_pre_label = torch.max(y_pre, 1)
    y_pre_label = y_pre_label.cpu().detach().numpy()

    y_label[np.arange(y_true.shape[0]), y_pre_label] = 1
    y_prob = y_pre.cpu().detach().numpy()
    y_pre = y_pre.cpu().detach().numpy() >= threshold


    y_label = y_label + y_pre
    y_label[y_label > 1.1] = 1

    labels = y_true
    binary_outputs = y_label


    # Define the weights, the SNOMED CT code for the normal class, and equivalent SNOMED CT codes.
    weights_file = '/scratch/physionet-2020/weights.csv'
    normal_class = '426783006'

    # Get the label
    label_file_dir = '/scratch/physionet-2020/dx_mapping_scored.csv'
    label_file = pd.read_csv(label_file_dir)
    equivalent_classes = ['59118001', '63593006', '17338001']
    classes = sorted(list(set([str(name) for name in label_file['SNOMED CT Code']]) - set(equivalent_classes)))
    

    weights = load_weights(weights_file, classes)

    # Only consider classes that are scored with the Challenge metric.
    indices = np.any(weights, axis=0) # Find indices of classes in weight matrix.
    classes = [x for i, x in enumerate(classes) if indices[i]]
    labels = labels[:, indices]
    binary_outputs = binary_outputs[:, indices]
    weights = weights[np.ix_(indices, indices)]

    challenge_metric = compute_challenge_metric(weights, labels, binary_outputs, classes, normal_class)

    # Return the results.
    return challenge_metric

# Compute macro AUROC and macro AUPRC.
def compute_auc(labels, outputs):
    
    num_recordings, num_classes = np.shape(labels)
    outputs = outputs.cpu().detach().numpy().astype(np.int)
    labels = labels.cpu().detach().numpy().astype(np.int)
    # Compute and summarize the confusion matrices for each class across at distinct output values.
    auroc = np.zeros(num_classes)
    auprc = np.zeros(num_classes)

    for k in range(num_classes):
        # We only need to compute TPs, FPs, FNs, and TNs at distinct output values.
        thresholds = np.unique(outputs[:, k])
        thresholds = np.append(thresholds, thresholds[-1]+1)
        thresholds = thresholds[::-1]
        num_thresholds = len(thresholds)

        # Initialize the TPs, FPs, FNs, and TNs.
        tp = np.zeros(num_thresholds)
        fp = np.zeros(num_thresholds)
        fn = np.zeros(num_thresholds)
        tn = np.zeros(num_thresholds)
        fn[0] = np.sum(labels[:, k]==1)
        tn[0] = np.sum(labels[:, k]==0)

        # Find the indices that result in sorted output values.
        idx = np.argsort(outputs[:, k])[::-1]

        # Compute the TPs, FPs, FNs, and TNs for class k across thresholds.
        i = 0
        for j in range(1, num_thresholds):
            # Initialize TPs, FPs, FNs, and TNs using values at previous threshold.
            tp[j] = tp[j-1]
            fp[j] = fp[j-1]
            fn[j] = fn[j-1]
            tn[j] = tn[j-1]

            # Update the TPs, FPs, FNs, and TNs at i-th output value.
            while i < num_recordings and outputs[idx[i], k] >= thresholds[j]:
                if labels[idx[i], k]:
                    tp[j] += 1
                    fn[j] -= 1
                else:
                    fp[j] += 1
                    tn[j] -= 1
                i += 1

        # Summarize the TPs, FPs, FNs, and TNs for class k.
        tpr = np.zeros(num_thresholds)
        tnr = np.zeros(num_thresholds)
        ppv = np.zeros(num_thresholds)
        npv = np.zeros(num_thresholds)

        for j in range(num_thresholds):
            if tp[j] + fn[j]:
                tpr[j] = float(tp[j]) / float(tp[j] + fn[j])
            else:
                tpr[j] = float('nan')
            if fp[j] + tn[j]:
                tnr[j] = float(tn[j]) / float(fp[j] + tn[j])
            else:
                tnr[j] = float('nan')
            if tp[j] + fp[j]:
                ppv[j] = float(tp[j]) / float(tp[j] + fp[j])
            else:
                ppv[j] = float('nan')

        #print("Sensitivity(tpr):",tpr," Specificity:",1-tnr)
        # Compute AUROC as the area under a piecewise linear function with TPR/
        # sensitivity (x-axis) and TNR/specificity (y-axis) and AUPRC as the area
        # under a piecewise constant with TPR/recall (x-axis) and PPV/precision
        # (y-axis) for class k.
        for j in range(num_thresholds-1):
            auroc[k] += 0.5 * (tpr[j+1] - tpr[j]) * (tnr[j+1] + tnr[j])
            auprc[k] += (tpr[j+1] - tpr[j]) * ppv[j+1]

    # Compute macro AUROC and macro AUPRC across classes.
    #print("auroc shape:",auroc.shape,"\nauroc:",auroc) #Class-wise aucroc
    macro_auroc = np.nanmean(auroc)
    macro_auprc = np.nanmean(auprc)

    return auroc, auprc

class optim_genetics:
    def __init__(self, target, outputs, classes):
        self.target = target
        self.outputs = outputs
        weights_file = '/scratch/physionet-2020/weights.csv'
        self.normal_class = '426783006'
        equivalent_classes = [['713427006', '59118001'],
                              ['284470004', '63593006'],
                              ['427172004', '17338001']]

        # Load the scored classes and the weights for the Challenge metric.
        self.weights = load_weights(weights_file, classes)
        self.classes = classes
        # match classed ordering
        # reorder = [self.classes.index(c) for c in classes]
        # self.outputs = self.outputs[:, reorder]
        # self.target = self.target[:, reorder]
        stop = 1

    def __call__(self, x):
        outputs = copy.deepcopy(self.outputs)
        outputs = outputs > x
        outputs = np.array(outputs, dtype=int)
        return -compute_challenge_metric(self.weights, self.target, outputs, self.classes, self.normal_class)

def find_thresholds(t,y, class_codes):

    N = 24
    f1prcT = np.zeros((N,))
    f1rocT = np.zeros((N,))

    for j in range(N):
        prc, rec, thr = precision_recall_curve(y_true=t[:, j], probas_pred=y[:, j])
        fscore = 2 * prc * rec / (prc + rec)
        idx = np.nanargmax(fscore)
        f1prc = np.nanmax(fscore)
        f1prcT[j] = thr[idx]

        fpr, tpr, thr = roc_curve(y_true=t[:, j], y_score=y[:, j])
        fscore = 2 * (1 - fpr) * tpr / (1 - fpr + tpr)
        idx = np.nanargmax(fscore)
        f1roc = np.nanmax(fscore)
        f1rocT[j] = thr[idx]

    population = np.random.rand(300, N)
    for i in range(1, 99):
        population[i, :] = i / 100

#     print(f1prcT)
#     print(f1rocT)
    population[100] = f1rocT
    population[101] = f1prcT
    bounds = [(0, 1) for i in range(N)]
    print('Differential_evolution started...')
    result = differential_evolution(optim_genetics(t, y, class_codes), bounds=bounds, disp=True, init=population, workers=-1)
    print('Differential_evolution ended...')
#     print(result)
    return result.x



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

print(device_count)

# Define the learning rate decay

def get_optimizer(opt,model,lr,momentum,weight_decay):
    
    if opt == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
                                   momentum=momentum, weight_decay=weight_decay,nestrov = True)
    elif opt == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
                                    weight_decay=weight_decay)
    else:
        raise Exception("optimizer not implement")
    
    return optimizer

# Define the learning rate decay
def get_lr_scheduler(lr_scheduler,optimizer,steps,gamma):
    
    if lr_scheduler == 'step':
        steps_list = [int(step) for step in steps.split(',')]
        #print(steps_list)
        lr_scheduler_fn = optim.lr_scheduler.MultiStepLR(optimizer, steps_list, gamma=gamma)
    elif lr_scheduler == 'exp':
        lr_scheduler_fn = optim.lr_scheduler.ExponentialLR(optimizer, gamma)
    elif lr_scheduler == 'stepLR':
        steps = int(steps)
        lr_scheduler_fn = optim.lr_scheduler.StepLR(optimizer, steps, gamma)
    elif lr_scheduler == 'cos':
        steps = int(steps)
        lr_scheduler_fn = optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, 0)
    elif lr_scheduler == 'fix':
        lr_scheduler_fn = None
    else:
        raise Exception("lr schedule not implement")
        
    return lr_scheduler_fn

# Define the monitoring accuracy
def accuracy_measuring_fn(monitor_acc):
    
    if monitor_acc == 'acc':
        cal_acc = None
    elif monitor_acc == 'AUC':
        cal_acc = RocAucEvaluation
    elif monitor_acc == 'ecgAcc':
        cal_acc = cal_Acc
    else:
        raise Exception("monitor_acc is not implement")
        
    return cal_acc

def print_confusion_matrix(confusion_matrix, axes, class_label,class_count, class_names, fontsize=14):

    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )

    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, ax=axes)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    axes.set_ylabel('True label')
    axes.set_xlabel('Predicted label')
    axes.set_title("Confusion Matrix for the class - {} [{}]".format(class_label,class_count))

def plot_graphs(history,epochs):

    epoch_list = [i for i in range(epochs)]
    logs = list(history.keys())
    fig = plt.figure(figsize=(10, 10))
    rows = 2
    columns = 2
    grid = plt.GridSpec(rows, columns, wspace = .25, hspace = .50)
    for i in range(rows*columns):
        #exec (f"plt.subplot(grid{[i]})")
        if i <=5:
            plt.plot(epoch_list, history[logs[i]])
            plt.title(logs[i])
    plt.show()

def get_predictions(labels,predictions,threshold):
    
    # Generate the outputs using the threshold
    labels_numpy = labels.cpu().detach().numpy().astype(np.int)
    
    y_label = np.zeros(labels_numpy.shape) #creating a dummy array

    _, y_pre_label = torch.max(predictions, 1)
    y_pre_label = y_pre_label.cpu().detach().numpy()

    y_label[np.arange(labels_numpy.shape[0]), y_pre_label] = 1
    y_prob = predictions.cpu().detach().numpy()
    y_pre = predictions.cpu().detach().numpy() >= threshold

    y_predictions = y_pre + y_label
    y_predictions[y_predictions > 1.1] = 1
    
    return y_predictions


def visulaise_performance(ground_truth,predictions,history,epochs,threshold,class_names,classwise_sample_count):
    
    
    challenge_metric = cal_Acc(ground_truth, predictions, threshold, num_classes=len(class_names))
    print('Final Challenge Metric Score',challenge_metric)
    
    #plot_graphs(history,epochs)
    auroc,auprc = compute_auc(ground_truth, predictions)
    
    #fig = go.Figure(data=[go.Table(header=dict(values=['classes','AUROC', 'AUPRC']),
    #             cells=dict(values=[class_names, auroc,auprc]))
    #                 ])
    #fig.show()

    
    preds = get_predictions(ground_truth,predictions,threshold)
    
    report = skm.classification_report(ground_truth.cpu().detach().numpy(),preds, target_names = class_names)
    
    print(report)
    
    cm = skm.multilabel_confusion_matrix(ground_truth.cpu().detach().numpy(), preds)
    
    print(cm)
    
    #fig, ax = plt.subplots(6, 4, figsize=(22, 10))
    
    #for axes, cfs_matrix, label, class_count in zip(ax.flatten(), cm, class_names, classwise_sample_count):
    #    print_confusion_matrix(cfs_matrix, axes, label,class_count, ["N", "Y"])

    #fig.tight_layout()
    #plt.show()

def train(criterion,optimizer,model,epoch,dataloader):
    
    model.train()
    
    # Define the temp variable
    epoch_start = time.time()
    epoch_loss = 0.0

    for batch_idx, (inputs, labels, age_gender,input_indices) in (enumerate(dataloader)):
        

        inputs = inputs.to(device)
        labels = labels.to(device)
        age_gender = age_gender.to(device)
        
        # Do the learning process, in val, we do not care about the gradient for relaxing
        with torch.set_grad_enabled(True):
            

            logits = model(inputs, age_gender) #check outputs
            sigmoid = nn.Sigmoid()
            logits_prob = sigmoid(logits)
            #print(logits_prob.shape)

            #saving the predictions and label
            if batch_idx == 0:
                labels_all = labels
                predictions_all = logits_prob
            else:
                labels_all = torch.cat((labels_all, labels), 0)
                predictions_all = torch.cat((predictions_all, logits_prob), 0)

            if 'Focal' in str(criterion):
                loss = criterion(logits,labels,dataloader.dataset.classwise_sample_count,len(dataloader.dataset))
            else:
                loss = criterion(logits,labels)
                
            loss_temp = loss.item() * inputs.size(0)
            epoch_loss += loss_temp

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

  
            
    train_auprc = average_precision_score(y_true=labels_all.cpu().detach().numpy(), y_score=predictions_all.cpu().detach().numpy())
    
    # Print the train and val information via each epoch
    epoch_loss = epoch_loss / len(dataloader)
    print('Training: Epoch: {} train-Loss: {:.4f} train-auprc {} Cost {:.1f} sec'.format(epoch, epoch_loss, train_auprc, time.time() - epoch_start))
    
    return epoch_loss,train_auprc

def validation(criterion, model, epoch, valid_dataloader,batch_size=16):
    
        
        model.eval()
        epoch_loss = 0
        dummy_input = torch.randn(16, 12, 5000, dtype=torch.float).to(device)
        age_dummy_input = torch.randn(16, 5, dtype=torch.float).to(device)
        total_time  = 0
        
        #GPU-WARM-UP: will automatically initialize the GPU and prevent it from going into power-saving mode when we measure time
        for _ in range(10):
            _ = model(dummy_input,age_dummy_input)

        
        with torch.no_grad():
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

            for batch_idx,(inputs ,labels,age_gender, _) in enumerate(valid_dataloader):
                
                inputs = inputs.to(device)
                labels = labels.to(device)
                age_gender = age_gender.to(device)
                
                starter.record()
                logits = model(inputs,age_gender)
                sigmoid = nn.Sigmoid()
                logits_prob = sigmoid(logits)
                ender.record()
                
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)/1000 #Returns time elapsed in milliseconds
                total_time += curr_time
                
                if 'Focal' in str(criterion):
                    loss = criterion(logits,labels,valid_dataloader.dataset.classwise_sample_count,len(valid_dataloader.dataset))
                else:
                    loss = criterion(logits,labels)
                loss_temp = loss.item() * inputs.size(0)
                epoch_loss += loss_temp
                

                
                #storing all the predictions
                if batch_idx == 0:  
                    labels_all = labels
                    predictions_all = logits_prob
                else:
                    labels_all = torch.cat((labels_all,labels), 0)
                    predictions_all = torch.cat((predictions_all, logits_prob), 0)
                
                    
                
                    
        epoch_loss = epoch_loss / len(valid_dataloader)
        
        #This formula gives the number of examples our network can process in one second.
        Throughput = (batch_size)/total_time
        

        
        valid_auprc = average_precision_score(y_true=labels_all.cpu().detach().numpy(), y_score=predictions_all.cpu().detach().numpy())

        print("Validation: loss:",epoch_loss," valid_auprc:",valid_auprc," total_time: ",total_time," secs")
        
        
        
        return labels_all,predictions_all,epoch_loss,valid_auprc

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

def setting_params(idx,opt,momentum,criterion,weight_decay,lr_scheduler,lr,steps,gamma,device):
    
    datasets = {}
    datasets['train'], datasets['val'] = ECG("/scratch/physionet2020_5Fold_Data/", str(idx)).data_preprare()
    train_dataloader = DataLoader(datasets['train'],batch_size=batch_size,shuffle = True)
    valid_dataloader = DataLoader(datasets['val'],batch_size=batch_size,shuffle = True)
    
    model = resnet18()

    model.to(device)
    
    optimizer = get_optimizer(opt,model,lr,momentum,weight_decay)
    #lr_scheduler_fn = get_lr_scheduler(lr_scheduler,optimizer,steps,gamma)
    lr_scheduler_fn = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_dataloader), epochs=max_epoch)
    cal_acc = accuracy_measuring_fn(monitor_acc)
    threshold = np.load(weight_list[idx])['arr_0']
    
    
    runner(idx, model,train_dataloader,valid_dataloader,start_epoch,max_epoch, lr_scheduler_fn, lr,criterion,optimizer,threshold,save_dir)

#PARAMETERS

max_epoch = 10
no_folds = 1

# Load the checkpoint
start_epoch = 0

weight_list = ['/scratch/physionet2020/magic_weight0.npz', '/scratch/physionet2020/magic_weight1.npz', '/scratch/physionet2020/magic_weight2.npz','/scratch/physionet2020/magic_weight3.npz', '/scratch/physionet2020/magic_weight4.npz', '/scratch/physionet2020/magic_weight_avg.npz']
save_dir = "./"

# Optimizers & Loss fn

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
    setting_params(3,opt,momentum,criterion,weight_decay,lr_scheduler,lr,steps,gamma,device)
