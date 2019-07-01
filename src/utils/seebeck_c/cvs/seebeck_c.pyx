# cython: profile=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

## distutils: extra_compile_args = -fopenmp
## distutils: extra_link_args = -fopenmp

import cython
import numpy as np
cimport numpy as np
from Seebeck.utils.matrice_mu_c cimport param
from Seebeck.utils.matrice_mu_c import MatriceMu
from Seebeck.utils.matrice_mu_c cimport MatriceMu
import multiprocessing
from Seebeck.src.system import System
import concurrent.futures as concfut
from libc cimport math
from cython.parallel cimport prange

cdef class Seebeck:
    def __init__(self, parametres=[], g3=[], temperatures=[], energies=[]):
        self.temperatures = temperatures
        self.energies = energies
        self.MatMu = MatriceMu()
        p = parametres.copy()
        p.update({"v":0.0, "E":0.0, "T":0.0, "beta":0.0})
        self.parametres = <param>p
        self.g3 = g3
        # self.sup = self.parametres.Np // 2
        # self.inf = -self.sup
        # self._set_mu(sys)
        # self.g3 = sys.g3
        # self.seebeck = {}
        # self.T = None
        # self.E = None
        # self.scattering_time = {T: {} for T in self.temperatures}


    cdef double[:,:,:] get_mu(self, double t, double e):
        self.parametres.T = t
        self.parametres.E = e
        self.MatMu.initialisation(self.parametres)
        return self.MatMu.calcul_matrice_mu()



    cdef double[:,:] get_collision_matrix(self, double t, double e):
        cdef:
            long k1
            int N = self.parametres.Np
            double[:,:] collision_matrix
            double[:,:,:] mu_2
            double[:,:,:] g3
        mu_2 = self.get_mu(t, e)
        g3 = self.g3[t]
        collision_matrix = np.empty([N, N], dtype=np.double)
        for k1 in range(N) :
            collision_matrix[k1] = self.get_row_collision_matrix(mu_2, k1, g3)

        collision_matrix = np.array(collision_matrix) + np.array(collision_matrix).T - \
            np.diag(np.diag(collision_matrix))

        return collision_matrix

    cdef double[:] get_row_collision_matrix(self, double[:,:,:] mu_2
                                                , long k1, double [:,:,:] g3) :
        cdef:
            #np.ndarray[double, ndim=1] row_col_matrix
            double[:] row_col_matrix
            long k2, k3, k4, i
            int N = self.parametres.Np
            double g3_1=0.0, g3_2=0.0, g3_3=0.0, S2_1=0.0, S2_2=0.0

        row_col_matrix = np.zeros([N],dtype=np.double)
        row_col_matrix[k1] = 0.0
        for k3 in range(N):  # self.inf,self.sup
            for k4 in range(N):
                g3_1=0.0
                i = (k3 + k4 - k1)#%N
                #i = (k3 + k4 - k1)
                if (0 <= i < N):
                    g3_1 = abs(g3[k1, i, k3] - g3[k1, i, k4]) ** 2
                row_col_matrix[k1] += mu_2.T[k1, k3, k4] * g3_1

        for k2 in range(k1+1,N):
            #if k2 != k1:
            row_col_matrix[k2] = 0.0
            for k3 in range(N):
                g3_2=0.0;g3_3=0.0
                S2_1=0.0; S2_2=0.0
                k4 = (k1 + k2 - k3)#%N
                if (0 <= k4 < N):
                    S2_1 = mu_2[k1, k2, k3]
                    g3_2 = abs(g3[(k1, k2, k3)] - g3[(k1, k2, k4)]) ** 2
                k4 = (k1 + k3 - k2)#%N  # + self.sup) % self.N - self.sup
                if (0 <= k4 < N):
                    S2_2 = mu_2[k1, k3, k2]
                    g3_3 = abs(g3[(k1, k3, k2)] - g3[(k1, k3, k4)]) ** 2
                row_col_matrix[k2] += (g3_2 * S2_1 - 2 * g3_3 * S2_2)
        return row_col_matrix

    #def test_set_mu(self, double T,double E):
    def test_set_mu(self, T,E):
        #mu = self.get_mu(T, E)
        #c = self.get_collision_matrix(T,E)
        #return mu, c
        return E % T
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

    def get_value_scattening_time(self,double[:,:] L, double t, double e):
        cdef:
            double s, err
        s, err = self.scatering_time(L, t, e)
        s /= math.cosh(e / 2.0 / t)
        return s, err



    def get_scattering_time(self, double t, double e):
        cdef:
            double[2] tau
            double[:,:] L
        L = self.get_collision_matrix(t, e)
        tau[0], tau[1] = self.get_value_scattening_time(L, t, e)
        return tau

    def get_seebeck_coefficient(self, double t):
        cdef:
            double s1, s2, e
            double err1, err2
            double[:,:] L
        e = -0.01
        L = self.get_collision_matrix(t, e)
        s1, err1 = self.scatering_time(L, t, e)
        e = 0.01
        L = self.get_collision_matrix(t, e)
        s2, err2 = self.scatering_time(L, t, e)
        seebeck = -0.5*math.log(s2/s1)* math.pi**3 * t / e
        return seebeck, max(err1, err2)



        # cython: boundscheck=False
        # cython: wraparound=False
        # cython: cdivision=True
        # cython: nonecheck=False
        #
        # cython: profile=True
