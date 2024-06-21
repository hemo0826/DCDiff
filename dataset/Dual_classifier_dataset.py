import torch
import glob
import random
import numpy as np
import pandas
from PIL import Image
import torchvision.transforms as transforms
class ClassifierDataset(torch.utils.data.Dataset):
    def __init__(self, path, mode='train', level=0.0):
        super().__init__()
        self.mode = mode
        self.data = []
        self.image = []
        self.label = []
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])

        if mode == 'train' or self.mode == 'val':
            filenames = sorted(glob.glob(path+str(level)+'/*/*'))


            random.seed(552)
            random.shuffle(filenames)
            random.seed()
            train_frac, val_frac = 0.7, 0.3
            n_train = int(train_frac*len(filenames)+1)

            n_val = int(len(filenames)-n_train)
            
            if mode == 'train':
                for i in range(n_train):
                    fname = filenames[i]


                    npy = np.load(fname)

                    self.data.append(npy)

                    label = fname.split("/")[-2].split('_')[0]

                    if label == 'tumor':
                        self.label.append(1)
                    if label == 'normal':
                        self.label.append(0)

            if mode == 'val':
                for i in range(n_train, n_train+n_val):
                    fname = filenames[i]


                    npy = np.load(fname)


                    self.data.append(npy)


                    label = fname.split('/')[-2].split('_')[0]

                    if label == 'tumor':
                        self.label.append(1)
                    if label == 'normal':
                        self.label.append(0)

        if mode == 'test':
            filenames = sorted(glob.glob(path+str(level)+'/*/*'))
            for fname in filenames:



                npy = np.load(fname)

                self.data.append(npy)

                case = fname.split('/')[-2]

                labels = pandas.read_csv(path + '/reference.csv', index_col=0, header=None)
                label = labels.loc[case, 1]

                if label == 'Tumor':
                    self.label.append(1)
                if label == 'Normal':
                    self.label.append(0)

    def __len__(self):
        return len(self.label)

    def augment(self, feats):
        np.random.shuffle(feats)
        return feats

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def __getitem__(self, index):

        return self.augment(self.data[index][0]), self.data[index][1], self.data[index][2],  self.label[index]
