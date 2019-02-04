import pandas as pd

# Param world
train_imgs = "/media/asilla/data102/hana/whale_pure/train/"
test_imgs = "/media/asilla/data102/hana/whale_pure/test/"
train_csv = "/media/asilla/data102/hana/whale_pure/train.csv"

# Splited csv
custom_train_csv = "./data/custom_train.csv"
custom_test_csv = "./data/custom_test.csv"

oversampled_csv = './oversampled_train_and_val.csv'
num_worker = 20
num_gpus = 2

train = pd.read_csv(custom_train_csv)
train = train.loc[train['Id'] != 'new_whale']

train_oversample = pd.read_csv(oversampled_csv)
train_oversample = train_oversample.loc[train_oversample['Id'] != 'new_whale']

val = pd.read_csv(custom_test_csv)
val = val.loc[train['Id'] != 'new_whale']

num_classes = len(train['Id'].unique())
