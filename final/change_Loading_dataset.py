import pandas as pd



from torch.utils.data import Dataset, DataLoader
import torch


from preprocess_data import Compose, Filtering, ZScore, NaNvalues, ValClip, RandomClip, Normalize, Retype
from preprocessing_data import load_data, prepare_data

class dataset(Dataset):

    def __init__(self, anno_pd, test=False, transform=None,class_codes=None, class_names=None, data_dir=None, loader=load_data):
        self.test = test
        if self.test:
            self.data = anno_pd['filename'].tolist()
            self.fs = anno_pd['fs'].tolist()
        else:
            self.data = anno_pd['filename'].tolist()
           #print('DATA:', data_dir)
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



class ECG(object):
    


    def __init__(self, data_dir,score_path, split='0'):
        self.data_dir = data_dir
        self.split = split
        self.score_path = score_path
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
            codes_mapping = pd.read_csv(self.score_path)
            class_codes = train_pd.columns.values[4:]
            class_names = [codes_mapping[codes_mapping['SNOMED CT Code']==int(label)]['Dx'].values[0] for label in class_codes]
            #print('ECG:', self.data_dir)
            
            
            train_dataset = dataset(anno_pd=train_pd, transform=data_transforms['train'],class_codes=class_codes,class_names=class_names, data_dir=self.data_dir)
            val_dataset = dataset(anno_pd=val_pd, transform=data_transforms['val'],class_codes=class_codes,class_names = class_names, data_dir=self.data_dir)
            return train_dataset, val_dataset


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
