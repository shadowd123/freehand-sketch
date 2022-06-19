import numpy as np
from sklearn import svm
from read import read_data
import time

def SVM():
	

	C = [  1 ]
	kernel = ['rbf']
	N = [10000]
	
	for each_n in N:
		r = read_data(num= each_n)
		train, test,  train_label, test_label = r.get_data()
		for each_kernel in kernel:
			for each_c in C:
				start_time  = time.time()

				classifier = svm.SVC(C=each_c,kernel = each_kernel)
				classifier.fit(train, train_label)
				y_val_pred = classifier.predict(test)        
				num_correct = np.sum(y_val_pred == test_label)
				num_val = test.shape[0]
				accuracy = float(num_correct) / num_val

				end_time = time.time()
				print("trainset size: ", train.shape[0],"C: ",each_c," kernel: ", each_kernel ," total time: ", end_time-start_time, "accuracy: ", accuracy)

SVM()