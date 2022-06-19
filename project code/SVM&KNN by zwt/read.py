import numpy as np
import time
from sklearn import svm

npz_path = "D:\233\dataset"
animal = ["bear", "camel", "cat", "cow", "crocodile", "dog", "elephant", "flamingo", "giraffe", "hedgehog","horse", "kangaroo",\
		"lion", "monkey", "owl","panda","penguin", "pig", "raccoon", "rhinoceros", "sheep", "squirrel", "tiger","whale", "zebra"]

# num = 65000

#数据集形状：一个数据集2500个sketch，一个sketch有三行

class read_data():
	
	def __init__(self,num):
		self.test = []
		self.train = []
		self.valid = []
		self.test_label = []
		self.train_label = []
		self.valid_label = []

		self.num = 70000-num

		count = 0
		for a in animal:
			data = np.load("sketchrnn_"+a+".npz",allow_pickle=True, encoding='latin1')
			if count == 0:
				self.test = data['test']
				self.train = data['train'][self.num:]
				self.valid = data['valid']
				self.test_label = np.zeros(shape=(2500),dtype=int)
				self.train_label = np.zeros(shape=(70000-self.num),dtype=int)
				self.valid_label = np.zeros(shape=(2500),dtype=int)
			else:
				self.test = np.concatenate((self.test, data['test']))
				self.train = np.concatenate((self.train, data['train'][self.num:]))
				self.valid = np.concatenate((self.valid, data['valid']))
				self.test_label = np.concatenate((self.test_label, np.full((2500), count)))
				self.train_label = np.concatenate((self.train_label, np.full((70000-self.num), count)))
				self.valid_label = np.concatenate((self.valid_label, np.full((2500), count)))
			count += 1
		
		#reshape成1维向量
		self.train = self.train.reshape((self.train.shape[0], 28*28))
		self.test = self.test.reshape((self.test.shape[0], 28*28))
		self.valid = self.valid.reshape((self.valid.shape[0], 28*28))
		
		print("trainset shape: ",self.train.shape)
		print("testset shape: ",self.test.shape)
		print("validset shape: ",self.valid.shape)
		

		#shuffle trainset
		
		state = np.random.get_state()
		np.random.shuffle(self.train)
		np.random.set_state(state)
		np.random.shuffle(self.train_label)


	def get_data(self):
		return self.train, self.test, self.train_label, self.test_label
