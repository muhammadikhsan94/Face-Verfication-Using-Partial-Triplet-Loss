import os
import numpy as np
import pandas as pd
import cv2

path_csv_train = '/home/m433788/Thesis/data_asli/CSV/train.csv'
path_csv_val = '/home/m433788/Thesis/data_asli/CSV/val.csv'
path_csv_test = '/home/m433788/Thesis/data_asli/CSV/test.csv'

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
		id_1, _, _ = df['image_1'][i].split('@')
		id_1_ = os.path.splitext(df['image_1'][i])[0]
		id_2, _, _ = df['image_2'][i].split('@')
		id_2_ = os.path.splitext(df['image_2'][i])[0]
		

		for t in range(num_region):
			pair_data = {}
			pair_data['image_1'] = os.path.join(
				id_1, id_1_, '0' + str(t + 1) + '.jpg')
			pair_data['image_2'] = os.path.join(
				id_2, id_2_, '0' + str(t + 1) + '.jpg')
			pair_data['label'] = df['label'].iloc[i]
				
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
		id_1, _, _ = df['image_1'][i].split('@')
		id_1_ = os.path.splitext(df['image_1'][i])[0]
		id_2, _, _ = df['image_2'][i].split('@')
		id_2_ = os.path.splitext(df['image_2'][i])[0]

		for t in range(num_region):
			pair_data = {}
			pair_data['image_1'] = os.path.join(
				id_1, id_1_, '0' + str(t + 1) + '.jpg')
			pair_data['image_2'] = os.path.join(
				id_2, id_2_, '0' + str(t + 1) + '.jpg')
			pair_data['label'] = df['label'].iloc[i]

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
		id_1, _, _ = df['image_1'][i].split('@')
		id_1_ = os.path.splitext(df['image_1'][i])[0]
		id_2, _, _ = df['image_2'][i].split('@')
		id_2_ = os.path.splitext(df['image_2'][i])[0]

		for t in range(num_region):
			pair_data = {}
			pair_data['image_1'] = os.path.join(
				id_1, id_1_, '0' + str(t + 1) + '.jpg')
			pair_data['image_2'] = os.path.join(
				id_2, id_2_, '0' + str(t + 1) + '.jpg')
			pair_data['label'] = df['label'].iloc[i]

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


sub_4SR_test = list_data_4_sub(path_csv_test, '/home/m433788/Thesis/data_asli/CSV/4SR_test_1.csv',
								'/home/m433788/Thesis/data_asli/CSV/4SR_test_2.csv', '/home/m433788/Thesis/data_asli/CSV/4SR_test_3.csv',
								'/home/m433788/Thesis/data_asli/CSV/4SR_test_4.csv')

sub_6SR_test = list_data_6_sub(path_csv_test, '/home/m433788/Thesis/data_asli/CSV/6SR_test_1.csv',
								'/home/m433788/Thesis/data_asli/CSV/6SR_test_2.csv', '/home/m433788/Thesis/data_asli/CSV/6SR_test_3.csv',
								'/home/m433788/Thesis/data_asli/CSV/6SR_test_4.csv', '/home/m433788/Thesis/data_asli/CSV/6SR_test_5.csv', '/home/m433788/Thesis/data_asli/CSV/6SR_test_6.csv')

sub_9SR_test = list_data_9_sub(path_csv_test, '/home/m433788/Thesis/data_asli/CSV/9SR_test_1.csv',
							   '/home/m433788/Thesis/data_asli/CSV/9SR_test_2.csv', '/home/m433788/Thesis/data_asli/CSV/9SR_test_3.csv',
							   '/home/m433788/Thesis/data_asli/CSV/9SR_test_4.csv', '/home/m433788/Thesis/data_asli/CSV/9SR_test_5.csv', '/home/m433788/Thesis/data_asli/CSV/9SR_test_6.csv', '/home/m433788/Thesis/data_asli/CSV/9SR_test_7.csv', '/home/m433788/Thesis/data_asli/CSV/9SR_test_8.csv', '/home/m433788/Thesis/data_asli/CSV/9SR_test_9.csv')
