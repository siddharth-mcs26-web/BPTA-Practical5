from Heuristic import Heuristic
from Gonzalez_approximation import Approximation
import numpy as np
from sklearn.cluster import KMeans

data_size = 500
k = 5
data_set = []
for i in range(data_size):
  data_set.append([np.random.randint(0, 500), np.random.randint(0, 500)])

approximation_max_distance = Approximation(data_set, k)
print("Gonzalez approximatoin distance = " , approximation_max_distance)
heuristic_distance = Heuristic(data_set, k)
print("Heuristic approximation distance = ", heuristic_distance)

