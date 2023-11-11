import numpy as np


from scipy.stats import zscore
import random


from scipy.io import loadmat
from scipy.signal import resample
from scipy import signal

import torch




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

def load_and_clean_sub(path):
    label_test = pd.read_csv(path, sep="\t", names=list(range(0, 3)))
    label_test.columns = ['id', 'age', 'gender']  
    return label_test


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
