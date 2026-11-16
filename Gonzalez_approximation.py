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
                d.append(abs(data_set[i][0] - data_set[c][0]) + abs(data_set[i][1] - data_set[c][1]))
            min_distance.append(min(d))

        center_idx.append(np.argmax(min_distance))
        if(np.max(min_distance) < max_d):
            max_d = np.max(min_distance)

    return max_d