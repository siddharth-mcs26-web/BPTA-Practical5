import numpy as np

def Approximation(data_set, k):
    center_idx = []
    center_idx.append(np.random.randint(0, len(data_set)))
    for i in range(k - 1):
        min_distance = []
        max_d = 100000
        for i in range(len(data_set)):
            d = []
            for c in center_idx:
                d.append(np.linalg.norm(data_set[i] - data_set[c]))
            min_distance.append(min(d))

        center_idx.append(np.argmax(min_distance))
        if(np.max(min_distance) < max_d):
            max_d = np.max(min_distance)

    return max_d