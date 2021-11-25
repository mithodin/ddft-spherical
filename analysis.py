from typing import Tuple

import numpy
import numpy as np
from scipy import sparse


class Analysis:
    def __init__(self, dr: float, n: int):
        self.dr = dr
        self.n = n
        self.__init_weights()
        self.__init_weights_shell()

    def __init_weights(self):
        dr, n, weights = self.__init_integral()
        self.__init_divergence(dr, n, weights)
        self.__init_gradient(dr, n)

    def __init_integral(self):
        n = self.n
        dr = self.dr
        weights = (6 * np.arange(n, dtype=np.float64) ** 2 + 1) * 2.0
        weights[0] = 1
        weights[-1] = (6 * (n - 1) ** 2 - 4 * (n - 1) + 1)
        weights *= dr ** 3 * np.pi / 3
        self.weights = weights
        return dr, n, weights

    def __init_gradient(self, dr, n):
        grad_op = np.zeros((n, n), dtype=np.float64)
        grad_op[0, 0:3] = [-3. / 2., 2., -1. / 2.]
        grad_op[-1, n - 3:n] = [1. / 2., -2., 3. / 2.]
        for k in range(1, n - 1):
            grad_op[k, k - 1:k + 2] = [-1. / 2., 0, 1. / 2.]
        grad_op /= dr
        self._grad_op = sparse.csr_matrix(grad_op)

        fwd_grad_op = np.zeros((n, n), dtype=np.float64)
        fwd_grad_op[n-1, n-2:n] = [-1., 1.]
        for k in range(0, n-1):
            fwd_grad_op[k, k:k+2] = [-1., 1.]
        fwd_grad_op /= dr
        self._fwd_grad_op = sparse.csr_matrix(fwd_grad_op)

    def __init_divergence(self, dr, n, weights):
        div_op = np.zeros((n, n), dtype=np.float64)
        k = np.arange(n, dtype=np.float64)
        k_intermediate = ((k+0.5)**3 + (k+0.5)/4)**(1./3.) - k
        for k in range(0, n):
            if k > 0:
                div_op[k, k-1] = (k-1+k_intermediate[k-1])**2*(1-k_intermediate[k-1])
                div_op[k, k] = (k-1+k_intermediate[k-1])**2*k_intermediate[k-1]
            if k < n-1:
                div_op[k, k] -= (k+k_intermediate[k])**2*(1-k_intermediate[k])
                div_op[k, k+1] = -(k+k_intermediate[k])**2*k_intermediate[k]
            else:
                div_op[k, k] -= (k+k_intermediate[k])**2
        div_op *= -4*np.pi*dr**2/(weights.reshape(-1, 1))
        self._div_op = sparse.csr_matrix(div_op)

    def __init_weights_shell(self):
        self.weights_shell = np.ones(self.n - 1)
        self.weights_shell *= self.dr

    def integrate(self, f: np.ndarray):
        return np.sum(f * self.weights)

    def integrate_shell(self, f: np.ndarray):
        return np.sum(f * self.weights_shell)

    def gradient(self, f: np.ndarray):
        return self._grad_op.dot(f)

    def forward_gradient(self, f: np.ndarray):
        return self._fwd_grad_op.dot(f)

    def delta(self) -> np.ndarray:
        res = np.zeros(self.n)
        res[0] = 1.0/self.weights[0]
        return res

    def divergence(self, f: np.ndarray) -> np.ndarray:
        return self._div_op.dot(f)

    def extrapolate(self, f: np.ndarray, fitrange: Tuple[int, int], extrapolate: Tuple[int, int])\
            -> np.ndarray:
        offset = fitrange[1] - 1
        x_fit = np.arange(*fitrange) - offset
        x_extrapolate = np.arange(*extrapolate) - offset
        slope = (x_fit*(f[fitrange[0]:fitrange[1]] - f[fitrange[1]-1])).sum() / (x_fit**2).sum()
        f[extrapolate[0]:extrapolate[1]] = f[offset] + slope * x_extrapolate
        return f
