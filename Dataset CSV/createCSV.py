import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

from itertools import combinations
from random import randrange

COMPARISON = 1

def get_files(images_path=''):
    list_files, list_labels = [], []

    # r=root, d=directories, f = files
    for r, d, f in os.walk(images_path):
        data_length = len(f)
        index = 0
        while index < data_length:
            list_id = []
            list_files.append(f[index])
            list_labels.append(d)
    return list_files, list_labels


data, label = get_files('G:/UPLOAD GDRIVE/Tanpa Kacamata/Dataset')
print(data.shape)


#def test_data(images_path='', save_path=''):
#    data = []
#    list_files, list_labels = get_files(images_path=images_path)

#    for index, files in enumerate(list_files):
#        comb = combinations(files, 2)
#        list_comb = list(comb)

#        for pairs in list_comb:
#            pair_data = {}
#            img1, img2 = pairs

#            pair_data['image_1'] = img1
#            pair_data['image_2'] = img2
#            pair_data['label'] = 0

#            data.append(pair_data)

#        count = 0
#        while count < len(list_comb) * COMPARISON:
#            for file in files:
#                pair_id = randrange(0, len(list_files))
#                while pair_id == index:
#                    pair_id = randrange(0, len(list_files))

#                pair_file = randrange(0, len(list_files[pair_id]))
#                pair_data = {}
#                pair_data['image_1'] = file
#                pair_data['image_2'] = list_files[pair_id][pair_file]
#                pair_data['label'] = 1
#                data.append(pair_data)
#                count += 1

#    df = pd.DataFrame(data)
#    df = df.sample(n=20000, random_state=1)
#    df.to_csv(save_path, index=False)


#def triplet_data(images_path='', save_path_train='', save_path_val=''):
#    data = []
#    list_files, list_labels = get_files(images_path=images_path)
#    list_labels = label_encoder.fit_transform(list_labels)

#    for index, files in enumerate(list_files):
#        comb = combinations(files, 2)
#        list_comb = list(comb)

#        for pairs in list_comb:
#            y = []
#            pair_data = {}
#            img1, img2 = pairs

#            pair_data['anchor'] = img1
#            pair_data['positif'] = img2

#            pair_id = randrange(0, len(list_files))
#            while pair_id == index:
#                pair_id = randrange(0, len(list_files))

#            pair_file = randrange(0, len(list_files[pair_id]))
#            pair_data['negatif'] = list_files[pair_id][pair_file]
#            pair_data['kelas'] = [list_labels[index], list_labels[index], list_labels[pair_id]]

#            data.append(pair_data)

#    df = pd.DataFrame(data)
#    train, val = train_test_split(df, test_size=0.2)
#    train = train.sample(n=16000, random_state=1)
#    val = val.sample(n=4000, random_state=1)
#    train.to_csv(save_path_train, index=False)
#    val.to_csv(save_path_val, index=False)


#train = triplet_data(
#    "/home/m433788/Thesis/data_asli/dataset_asli/Training/", "/home/m433788/Thesis/data_asli/train.csv", "/home/m433788/Thesis/data_asli/val.csv")
#train = test_data(
#    "/home/m433788/Thesis/data_asli/dataset_asli/Testing/", "/home/m433788/Thesis/data_asli/test.csv")
