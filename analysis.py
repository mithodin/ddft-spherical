import numpy as np
from scipy import sparse


class Analysis:
    def __init__(self, dr: float, n: int):
        self.dr = dr
        self.n = n
        self.__init_weights()

    def __init_weights(self):
        n = self.n
        dr = self.dr

        self.weights = (6 * np.arange(n) ** 2 + 1) * 2.0
        self.weights[0] = 1
        self.weights[-1] = (6 * (n - 1) ** 2 - 4 * (n - 1) + 1)
        self.weights *= dr ** 3 * np.pi / 3

        weights = self.weights
        div_op = np.zeros((n, n))
        div_op[0, 1] = -weights[0]
        div_op[-1, -2] = (n-2)**2/weights[-1]
        div_op[-1, -1] = -n**2/weights[-1]
        for k in range(1, n-1):
            div_op[k, k-1] = (k-1)**2/weights[k]
            div_op[k, k+1] = -(k+1)**2/weights[k]
        div_op *= -2*np.pi*dr**2
        self._div_op = sparse.csr_matrix(div_op)

    def integrate(self, f: np.array):
        return np.sum(f * self.weights)

    def gradient(self, f: np.array):
        res = (np.roll(f, -1) - np.roll(f, 1)) / 2.
        res[0] = (-3 * f[0] + 4 * f[1] - f[2]) / 2.
        res[-1] = (f[-3] - 4 * f[-2] + 3 * f[-1]) / 2.
        return res/self.dr

    def delta(self):
        res = np.zeros(self.n)
        res[0] = 1.0/self.weights[0]
        return res

    def divergence(self, f):
        return self._div_op.dot(f)
