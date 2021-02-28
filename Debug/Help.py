import numpy as np

data = [[[1, 2], [2, 3], [1, 3]],
        [[1, 4], [4, 3], [5, 2]],
        [[7, 1], [7, 2], [7, 3]]]
data1 = [[[1, 2], [2, 3], [1, 3]],
         [[1, 4], [4, 3], [5, 2]]]

for i in range(10):
    data1.append([i, i + 1])
print(data1)

