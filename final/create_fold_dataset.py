import os
import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from tqdm import tqdm
import time
import copy


def lsdir(rootdir="", suffix=".png"):
    file_list = []
    #assert os.path.exists(rootdir)
    for r, y, names in os.walk(rootdir):
        for name in names:
            if str(name).endswith(suffix):
                file_list.append(os.path.join(r, name))
    return file_list


def get_class_pair(input_directory_name, CT_codes_all, files, class_df):
    i = -1
    for file in files:
        g = file.replace('.mat', '.hea')
        input_file_name = g
        #print('name:',input_file_name)
        flag = 1
        with open(input_file_name, 'r') as f:
            for lines in f:
                if lines.startswith('#Dx'):
                    tmp = lines.split(': ')[1].split(',')
                    tmp = [c.strip() for c in tmp]
                    if len(list(set(tmp).intersection(set(CT_codes_all)))) == 0:
                        flag = 0
        if flag == 1:
            i = i + 1
            class_df.loc[i, 'filename'] = file
            with open(input_file_name, 'r') as f:
                for k, lines in enumerate(f):
                    if k == 0:
                        tmp = lines.split(' ')[2].strip()
                        class_df.loc[i, 'fs'] = int(tmp)
                    if lines.startswith('#Age'):
                        tmp = lines.split(': ')[1].strip()
                        if tmp == 'NaN':
                            class_df.loc[i, 'age'] = -1
                        else:
                            class_df.loc[i, 'age'] = int(tmp)
                    if lines.startswith('#Sex'):
                        tmp = lines.split(': ')[1].strip()
                        if tmp == 'NaN':
                            class_df.loc[i, 'gender'] = 'Unknown'
                        else:
                            class_df.loc[i, 'gender'] = tmp
                    if lines.startswith('#Dx'):
                        tmp = lines.split(': ')[1].split(',')
                        for c in tmp:
                            c = c.strip()
                            if c in CT_codes_all:
                                class_df.loc[i, c] = 1
    # class_df = class_df.iloc[0:i+1, :]
    class_df = class_df.drop(class_df.index[i + 1:])
    return class_df


def data_split(df, labels, n_split):
    X = np.arange(labels.shape[0])
    mskf = MultilabelStratifiedKFold(n_splits=n_split, random_state=2020, shuffle=True)

    split_index_list = []
    for train_index, test_index in mskf.split(X, labels):
        split_index_list.append([train_index, test_index])
    train_csv = []
    test_csv = []
    for i in range(n_split):
        train_csv.append(df.iloc[split_index_list[i][0], :])
        test_csv.append(df.iloc[split_index_list[i][1], :])

    return train_csv, test_csv
	
	
def read_and_split_data(data_directory):
    save_dir = '/scratch/physionet2020_5Fold_Data'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    #print('data:',data_directory)
    # Get every file
    input_files = lsdir(rootdir=data_directory, suffix=".mat")
    #print('input file:',len(input_files))
    # print(input_files)
    # set different dataset
    dataset_names = ['A', 'Q', 'I', 'S', 'HR', 'E']

    # Get the label
    label_file_dir = '/scratch/physionet2020/dx_mapping_scored.csv'
    label_file = pd.read_csv(label_file_dir)
    CT_codes = sorted([str(name) for name in label_file['SNOMED CT Code']])

    print('split the data')
    # split the data
    dataset_train = []
    dataset_test = []
    split_number = 5
    equivalent_classes = [['713427006', '59118001'], ['284470004', '63593006'], ['427172004', '17338001']]
    for dataset_name in tqdm(dataset_names):
        input_files_tmp = [i for i in input_files if os.path.basename(i).startswith(dataset_name)]
        if len(input_files_tmp) > 0:
            # put all the samples into a dataframe
            columns_names = ['filename', 'age', 'gender', 'fs'] + CT_codes
            all_zeros = np.zeros((len(input_files_tmp), len(columns_names)))
            df_zeros = pd.DataFrame(all_zeros, columns=columns_names)
            class_df_all = get_class_pair(data_directory, CT_codes, input_files_tmp, df_zeros)
            
            # Deal with three equivalent pairs
            droplist = []
            for pair in equivalent_classes:
                class_df_all.loc[:, pair[0]] = class_df_all[pair[0]] + class_df_all[pair[1]]
                class_df_all.loc[(class_df_all[pair[0]] == 2), pair[0]] = 1
                droplist.append(pair[1])
            class_df_all = class_df_all.drop(columns=droplist, axis=1)
            train_labels = class_df_all.loc[:, (class_df_all.sum(axis=0) != 0)].iloc[:, 3:].values
            train_tmp, test_tmp = data_split(df=class_df_all, labels=train_labels, n_split=split_number)
            #print(train_tmp)
            dataset_train.append(train_tmp)
            dataset_test.append(test_tmp)

    print('combine different datasets and save them as .csv')
    print('dataset trian:', len(dataset_train))
    # combine different datasets and save them as .csv
    for i in tqdm(range(split_number)):
        data_split_train = dataset_train[0][i].copy()
        data_split_test = dataset_test[0][i].copy()
        for j in range(len(dataset_train)):
            if j > 0:
                data_split_train = pd.concat([data_split_train, dataset_train[j][i]], ignore_index=True)
                data_split_test = pd.concat([data_split_test, dataset_test[j][i]], ignore_index=True)

        data_split_train.to_csv(os.path.join(save_dir, '%s.csv' % ('train_split' + str(i))), sep=',', index=False)
        data_split_test.to_csv(os.path.join(save_dir, '%s.csv' % ('test_split' + str(i))), sep=',', index=False)
		
data_directory = '/scratch/physionet2020'
read_and_split_data(data_directory)
