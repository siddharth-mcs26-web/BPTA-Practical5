from Heuristic import Heuristic
from Gonzalez_approximation import Approximation
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.datasets import make_blobs
from sklearn.datasets import fetch_california_housing
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_wine
from sklearn.datasets import load_diabetes
import time

#X, y = make_blobs(n_features=3, n_samples=500, centers=3)
#data_set = load_iris().data
#data_set = fetch_california_housing().data
#data_set = load_breast_cancer().data
#data_set = load_wine().data
data_set = load_diabetes().data
data_set = StandardScaler().fit_transform(data_set)
k=3
start_time = time.perf_counter()
approximation_max_distance = Approximation(data_set, k)
end_time = time.perf_counter()
print("Gonzalez approximation time = ", round(end_time - start_time, 10))
print("Gonzalez approximatoin distance = " , approximation_max_distance)
start_time = time.perf_counter()
heuristic_distance = Heuristic(data_set, k)
end_time = time.perf_counter()
print("Heuristic approximation time = ", round(end_time - start_time, 10))
print("Heuristic approximation distance = ", heuristic_distance)

