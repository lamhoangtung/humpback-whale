# Param world
train_imgs = "./data/train/"
test_imgs = "./data/test/"
train_csv = "./data/train.csv"
train = pd.read_csv(train_csv)
train = train.loc[train['Id'] != 'new_whale']
num_classes = len(train['Id'].unique())
