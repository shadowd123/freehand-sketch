from read import read_data
from sklearn import neighbors
import numpy as np
import time

def KNN():
	N = [10000]
	Metric = [ 'euclidean', 'chebyshev','manhattan']
	
	for each_n in N:
		for each_m in Metric:
			r = read_data(num = each_n)
			train, test, train_label, test_label = r.get_data()

			start_time = time.time()
			knn_classifier = neighbors.KNeighborsClassifier(n_neighbors=25, metric= each_m)
			knn_classifier.fit(train, train_label)
			y_val_pred = knn_classifier.predict(test)        
			num_correct = np.sum(y_val_pred == test_label)
			num_val = test.shape[0]
			accuracy = float(num_correct) / num_val
			end_time = time.time()

			print("train_size: ", each_n ,"metric: ", each_m,"total time: ", end_time-start_time, "accuracy: ", accuracy)



KNN()