import pandas as pd

# Param world
train_imgs = "./data/train/"
test_imgs = "./data/test/"
train_csv = "./data/train.csv"
oversampled_csv = './oversampled_train_and_val.csv'
num_worker = 20

train = pd.read_csv(train_csv)
train = train.loc[train['Id'] != 'new_whale']
num_classes = len(train['Id'].unique())
