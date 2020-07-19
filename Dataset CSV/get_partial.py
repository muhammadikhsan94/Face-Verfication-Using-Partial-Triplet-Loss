import os
import numpy as np
import pandas as pd
import cv2

path_csv_train = '/home/m433788/Thesis/data_asli/train.csv'
path_csv_val = '/home/m433788/Thesis/data_asli/val.csv'

path_4_sub = '/home/m433788/Thesis/data_asli/4_subregion'
path_6_sub = '/home/m433788/Thesis/data_asli/6_subregion'
path_9_sub = '/home/m433788/Thesis/data_asli/9_subregion'
path_4_sub_overlapping = '/home/m433788/Thesis/data_asli/4_subregion_overlapping'
path_6_sub_overlapping = '/home/m433788/Thesis/data_asli/6_subregion_overlapping'
path_9_sub_overlapping = '/home/m433788/Thesis/data_asli/9_subregion_overlapping'

def list_data_4_sub(path_csv, save_path_1='', save_path_2='', save_path_3='', save_path_4=''):
    num_region = 4

    sub_1, sub_2, sub_3, sub_4, sub_5, sub_6, sub_7, sub_8, sub_9 = [
    ], [], [], [], [], [], [], [], []
    df = pd.read_csv(path_csv, header=0)

    for i in range(len(df)):
        id_anc, _, _ = df['anchor'][i].split('@')
        id_anc_ = os.path.splitext(df['anchor'][i])[0]
        id_pos, _, _ = df['positif'][i].split('@')
        id_pos_ = os.path.splitext(df['positif'][i])[0]
        id_neg, _, _ = df['negatif'][i].split('@')
        id_neg_ = os.path.splitext(df['negatif'][i])[0]

        for t in range(num_region):
            pair_data = {}
            pair_data['anchor'] = os.path.join(
                id_anc, id_anc_, '0' + str(t + 1) + '.jpg')
            pair_data['positif'] = os.path.join(
                id_pos, id_pos_, '0' + str(t + 1) + '.jpg')
            pair_data['negatif'] = os.path.join(
                id_neg, id_neg_, '0' + str(t + 1) + '.jpg')
            pair_data['kelas'] = df['kelas'].iloc[i]
                
            if t == 0:
                sub_1.append(pair_data)
            elif t == 1:
                sub_2.append(pair_data)
            elif t == 2:
                sub_3.append(pair_data)
            elif t == 3:
                sub_4.append(pair_data)
            elif t == 4:
                sub_5.append(pair_data)
            elif t == 5:
                sub_6.append(pair_data)
            elif t == 6:
                sub_7.append(pair_data)
            elif t == 7:
                sub_8.append(pair_data)
            else:
                sub_9.append(pair_data)
        
    df_1 = pd.DataFrame(sub_1)
    df_1.to_csv(save_path_1, index=False)
    df_2 = pd.DataFrame(sub_2)
    df_2.to_csv(save_path_2, index=False)
    df_3 = pd.DataFrame(sub_3)
    df_3.to_csv(save_path_3, index=False)
    df_4 = pd.DataFrame(sub_4)
    df_4.to_csv(save_path_4, index=False)


def list_data_6_sub(path_csv, save_path_1='', save_path_2='', save_path_3='', save_path_4='', save_path_5='', save_path_6=''):
    num_region = 6

    sub_1, sub_2, sub_3, sub_4, sub_5, sub_6 = [], [], [], [], [], []
    df = pd.read_csv(path_csv, header=0)

    for i in range(len(df)):
        id_anc, _, _ = df['anchor'][i].split('@')
        id_anc_ = os.path.splitext(df['anchor'][i])[0]
        id_pos, _, _ = df['positif'][i].split('@')
        id_pos_ = os.path.splitext(df['positif'][i])[0]
        id_neg, _, _ = df['negatif'][i].split('@')
        id_neg_ = os.path.splitext(df['negatif'][i])[0]

        for t in range(num_region):
            pair_data = {}
            pair_data['anchor'] = os.path.join(
                id_anc, id_anc_, '0' + str(t + 1) + '.jpg')
            pair_data['positif'] = os.path.join(
                id_pos, id_pos_, '0' + str(t + 1) + '.jpg')
            pair_data['negatif'] = os.path.join(
                id_neg, id_neg_, '0' + str(t + 1) + '.jpg')
            pair_data['kelas'] = df['kelas'].iloc[i]

            if t == 0:
                sub_1.append(pair_data)
            elif t == 1:
                sub_2.append(pair_data)
            elif t == 2:
                sub_3.append(pair_data)
            elif t == 3:
                sub_4.append(pair_data)
            elif t == 4:
                sub_5.append(pair_data)
            else:
                sub_6.append(pair_data)

    df_1 = pd.DataFrame(sub_1)
    df_1.to_csv(save_path_1, index=False)
    df_2 = pd.DataFrame(sub_2)
    df_2.to_csv(save_path_2, index=False)
    df_3 = pd.DataFrame(sub_3)
    df_3.to_csv(save_path_3, index=False)
    df_4 = pd.DataFrame(sub_4)
    df_4.to_csv(save_path_4, index=False)
    df_5 = pd.DataFrame(sub_5)
    df_5.to_csv(save_path_5, index=False)
    df_6 = pd.DataFrame(sub_6)
    df_6.to_csv(save_path_6, index=False)


def list_data_9_sub(path_csv, save_path_1='', save_path_2='', save_path_3='', save_path_4='', save_path_5='', save_path_6='', save_path_7='', save_path_8='', save_path_9=''):
    num_region = 9

    sub_1, sub_2, sub_3, sub_4, sub_5, sub_6, sub_7, sub_8, sub_9 = [
    ], [], [], [], [], [], [], [], []
    df = pd.read_csv(path_csv, header=0)

    for i in range(len(df)):
        id_anc, _, _ = df['anchor'][i].split('@')
        id_anc_ = os.path.splitext(df['anchor'][i])[0]
        id_pos, _, _ = df['positif'][i].split('@')
        id_pos_ = os.path.splitext(df['positif'][i])[0]
        id_neg, _, _ = df['negatif'][i].split('@')
        id_neg_ = os.path.splitext(df['negatif'][i])[0]

        for t in range(num_region):
            pair_data = {}
            pair_data['anchor'] = os.path.join(
                id_anc, id_anc_, '0' + str(t + 1) + '.jpg')
            pair_data['positif'] = os.path.join(
                id_pos, id_pos_, '0' + str(t + 1) + '.jpg')
            pair_data['negatif'] = os.path.join(
                id_neg, id_neg_, '0' + str(t + 1) + '.jpg')
            pair_data['kelas'] = df['kelas'].iloc[i]

            if t == 0:
                sub_1.append(pair_data)
            elif t == 1:
                sub_2.append(pair_data)
            elif t == 2:
                sub_3.append(pair_data)
            elif t == 3:
                sub_4.append(pair_data)
            elif t == 4:
                sub_5.append(pair_data)
            elif t == 5:
                sub_6.append(pair_data)
            elif t == 6:
                sub_7.append(pair_data)
            elif t == 7:
                sub_8.append(pair_data)
            else:
                sub_9.append(pair_data)

    df_1 = pd.DataFrame(sub_1)
    df_1.to_csv(save_path_1, index=False)
    df_2 = pd.DataFrame(sub_2)
    df_2.to_csv(save_path_2, index=False)
    df_3 = pd.DataFrame(sub_3)
    df_3.to_csv(save_path_3, index=False)
    df_4 = pd.DataFrame(sub_4)
    df_4.to_csv(save_path_4, index=False)
    df_5 = pd.DataFrame(sub_5)
    df_5.to_csv(save_path_5, index=False)
    df_6 = pd.DataFrame(sub_6)
    df_6.to_csv(save_path_6, index=False)
    df_7 = pd.DataFrame(sub_7)
    df_7.to_csv(save_path_7, index=False)
    df_8 = pd.DataFrame(sub_8)
    df_8.to_csv(save_path_8, index=False)
    df_9 = pd.DataFrame(sub_9)
    df_9.to_csv(save_path_9, index=False)


sub_4SR_train = list_data_4_sub(path_csv_train, '/home/m433788/Thesis/data_asli/4SR_train_1.csv',
                                '/home/m433788/Thesis/data_asli/4SR_train_2.csv', '/home/m433788/Thesis/data_asli/4SR_train_3.csv',
                                '/home/m433788/Thesis/data_asli/4SR_train_4.csv')


sub_4SR_val = list_data_4_sub(path_csv_val, '/home/m433788/Thesis/data_asli/4SR_val_1.csv',
                               '/home/m433788/Thesis/data_asli/4SR_val_2.csv', '/home/m433788/Thesis/data_asli/4SR_val_3.csv',
                               '/home/m433788/Thesis/data_asli/4SR_val_4.csv')

sub_6SR_train = list_data_6_sub(path_csv_train, '/home/m433788/Thesis/data_asli/6SR_train_1.csv',
                                '/home/m433788/Thesis/data_asli/6SR_train_2.csv', '/home/m433788/Thesis/data_asli/6SR_train_3.csv',
                                '/home/m433788/Thesis/data_asli/6SR_train_4.csv', '/home/m433788/Thesis/data_asli/6SR_train_5.csv', '/home/m433788/Thesis/data_asli/6SR_train_6.csv')


sub_6SR_val = list_data_6_sub(path_csv_val, '/home/m433788/Thesis/data_asli/6SR_val_1.csv',
                               '/home/m433788/Thesis/data_asli/6SR_val_2.csv', '/home/m433788/Thesis/data_asli/6SR_val_3.csv',
                               '/home/m433788/Thesis/data_asli/6SR_val_4.csv', '/home/m433788/Thesis/data_asli/6SR_val_5.csv', '/home/m433788/Thesis/data_asli/6SR_val_6.csv')


sub_9SR_train = list_data_9_sub(path_csv_train, '/home/m433788/Thesis/data_asli/9SR_train_1.csv',
                                '/home/m433788/Thesis/data_asli/9SR_train_2.csv', '/home/m433788/Thesis/data_asli/9SR_train_3.csv',
                                '/home/m433788/Thesis/data_asli/9SR_train_4.csv', '/home/m433788/Thesis/data_asli/9SR_train_5.csv', '/home/m433788/Thesis/data_asli/9SR_train_6.csv', '/home/m433788/Thesis/data_asli/9SR_train_7.csv', '/home/m433788/Thesis/data_asli/9SR_train_8.csv', '/home/m433788/Thesis/data_asli/9SR_train_9.csv')


sub_9SR_val = list_data_9_sub(path_csv_val, '/home/m433788/Thesis/data_asli/9SR_val_1.csv',
                               '/home/m433788/Thesis/data_asli/9SR_val_2.csv', '/home/m433788/Thesis/data_asli/9SR_val_3.csv',
                               '/home/m433788/Thesis/data_asli/9SR_val_4.csv', '/home/m433788/Thesis/data_asli/9SR_val_5.csv', '/home/m433788/Thesis/data_asli/9SR_val_6.csv', '/home/m433788/Thesis/data_asli/9SR_val_7.csv', '/home/m433788/Thesis/data_asli/9SR_val_8.csv', '/home/m433788/Thesis/data_asli/9SR_val_9.csv')
