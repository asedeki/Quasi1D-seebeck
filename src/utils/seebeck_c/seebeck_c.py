# cython: profile=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

## distutils: extra_compile_args = -fopenmp
## distutils: extra_link_args = -fopenmp

import numpy as np

from Seebeck.utils.matricediffusion import MatriceDiffusion
from Seebeck.src.system import System
import math


class Seebeck:
    def __init__(self, parametres=[], g3=[], temperatures=[], energies=[]):
        self.temperatures = temperatures
        self.energies = energies
        self.MatriceDiffusion = MatriceDiffusion()
        p = parametres.copy()
        p.update({"v": 0.0, "E": 0.0, "T": 0.0, "beta": 0.0})
        self.parametres = p
        self.g3 = g3

    def double[:, :, :] get_collision_matrix(self, double t, double e):
        self.parametres.T = t
        self.parametres.E = e
        self.MatriceDiffusion.initialisation(self.parametres, self.g3[t])
        return self.MatriceDiffusion.get_collision_matrix()

    def scatering_time(self, L, t, e):
        import scipy
        import scipy.linalg.lapack as lapack
        try:
            b = np.ones([self.parametres.Np])
            #lu, piv, x, info = lapack.dgesv(L, b)
            #x, resid, rank, s = scipy.linalg.lstsq(L, b)
            x = np.linalg.solve(L, b)
            err = np.linalg.norm(np.dot(L, x) - b)
            return sum(x), err
        except Exception as e:
            print(f"Probleme pour T={t}, E={e}")
            print(e)
            return np.nan, 10**10

    def get_value_scattening_time(self, double[:, :] L, double t, double e):
        cdef:
            double s, err
        s, err = self.scatering_time(L, t, e)
        s /= math.cosh(e / 2.0 / t)
        return s, err

    def get_scattering_time(self, double t, double e):
        cdef:
            double[2] tau
            double[:, :] L
        L = self.get_collision_matrix(t, e)
        tau[0], tau[1] = self.get_value_scattening_time(L, t, e)
        return tau

    def get_seebeck_coefficient(self, double t):
        cdef:
            double s1, s2, e
            double err1, err2
            double[:, :] L
        e = -0.01
        L = self.get_collision_matrix(t, e)
        s1, err1 = self.scatering_time(L, t, e)
        e = 0.01
        L = self.get_collision_matrix(t, e)
        s2, err2 = self.scatering_time(L, t, e)
        seebeck = -0.5*math.log(s2/s1) * math.pi**3 * t / e
        return seebeck, max(err1, err2)
