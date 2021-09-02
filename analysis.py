import numpy as np


class Analysis:
    def __init__(self, dr: float, n: int):
        self.dr = dr
        self.n = n
        self.__init_weights()

    def __init_weights(self):
        self.weights = (6 * np.arange(self.n) ** 2 + 1) * 2.0
        self.weights[0] = 1
        self.weights[-1] = (6 * (self.n - 1) ** 2 - 4 * (self.n - 1) + 1)
        self.weights *= self.dr ** 3 * np.pi / 3

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
        weights = self.weights
        dr = self.dr
        n = self.n
        fk2 = f*np.arange(n)**2
        div = (np.roll(fk2, 1) - np.roll(fk2, -1))/weights
        div[0] = -fk2[1]/weights[0]
        div[-1] = (fk2[-2] - fk2[-1]/(n-1)**2*n**2)/weights[-1]
        # there is a factor of /2 here that I do not entirely understand
        return -2*np.pi*dr**2*div
