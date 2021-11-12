import numpy as np


class GolayCode:

    def __init__(self):
        # число информационных (исходных) разрядов
        self.k = 12
        # длина закодированного сообщения
        self.n = 24
        # матрица B
        self.B = np.array([
            [1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1],
            [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1],
            [0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1],
            [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1],
            [1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1],
            [1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1],
            [0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1],
            [0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
            [0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1],
            [1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1],
            [0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
        ])
        self.G = np.concatenate([np.eye(self.k, dtype=int), self.B], axis=1)
        self.H = np.concatenate([np.eye(self.k, dtype=int), self.B], axis=0)