import scipy.io as sio
import numpy as np
import numpy.ma as ma
import torch
from torch.utils.data import Dataset


categories = ['cow', 'panda', 'lion', 'tiger', 'raccoon',
              'monkey', 'hedgehog', 'zebra', 'horse', 'owl',
              'elephant', 'squirrel', 'sheep', 'dog', 'bear',
              'kangaroo', 'whale', 'crocodile', 'rhinoceros',  'penguin',
              'camel', 'flamingo', 'giraffe', 'pig', 'cat']
categories2id = {categories[i]: i for i in range(len(categories))}

Num_cate = 25


class LoadData():
    def __init__(self):
        self.train_set = None
        self.valid_set = None
        self.test_set = None
        self.train_label = None
        self.valid_label = None
        self.test_label = None
        self.num_dataset = Num_cate
        l1m, l2m, l3m = 0, 0, 0
        for category in categories[:self.num_dataset]:
            filename = "dataset/sketchrnn_" + category + ".npz"
            load_data = np.load(filename, allow_pickle=True, encoding="latin1")
            train_data = load_data['train']
            valid_data = load_data['valid']
            test_data = load_data['test']

            for i in range(len(train_data)):
                l1 = train_data[i].shape[0]
                l1m = max(l1m, l1)
            for i in range(len(valid_data)):
                l2 = valid_data[i].shape[0]
                l2m = max(l2m, l2)
            for i in range(len(test_data)):
                l3 = test_data[i].shape[0]
                l2m = max(l3m, l3)
        self.l1m = l1m
        self.l2m = l2m
        self.l3m = l3m
        for category in categories[:self.num_dataset]:
            cid = categories2id[category]
            filename = "dataset/sketchrnn_" + category + ".npz"
            load_data = np.load(filename, allow_pickle=True, encoding="latin1")
            train_data = load_data['train']
            valid_data = load_data['valid']
            test_data = load_data['test']
            l1, l2, l3 = len(train_data), len(valid_data), len(test_data)
            train_label = np.full((l1, 1), cid, dtype=int)
            valid_label = np.full((l2, 1), cid, dtype=int)
            test_label = np.full((l3, 1), cid, dtype=int)

            if self.train_set is not None:
                self.train_set = np.concatenate((self.train_set, train_data))
                self.train_label = np.concatenate((self.train_label, train_label))
            else:
                self.train_set = train_data
                self.train_label = train_label
            if self.valid_set is not None:
                self.valid_set = np.concatenate((self.valid_set, valid_data))
                self.valid_label = np.concatenate((self.valid_label, valid_label))
            else:
                self.valid_set = valid_data
                self.valid_label = valid_label
            if self.test_set is not None:
                self.test_set = np.concatenate((self.test_set, test_data))
                self.test_label = np.concatenate((self.test_label, test_label))
            else:
                self.test_set = test_data
                self.test_label = test_label
            print(f"{category} dataset loaded from sketchrnn")

        print(f"{self.num_dataset} datasets loaded from sketchrnn")

    def get_data(self):
        return self.train_set, self.train_label, self.valid_set, self.valid_label, self.test_set, self.test_label

    def get_length(self):
        return self.l1m, self.l2m, self.l3m


class SketchRnnDataset(Dataset):
    def __init__(self, x, y, lm):
        self.x = x
        self.y = y
        self.lm = lm

    def __getitem__(self, index):
        li = len(self.x[index])
        x = np.resize(self.x[index], (self.lm, 3))
        x[li:, :] = 0
        x = torch.tensor(x).float()
        y = torch.tensor(self.y[index]).long()
        return x, y

    def __len__(self):
        return len(self.x)
