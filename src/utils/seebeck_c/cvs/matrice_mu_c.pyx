# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

# cython: profile=True

cimport cython
import numpy as np
cimport numpy as np
from libc cimport math

cdef class MatriceMu:
    """
        Calcul de la matrice Mu dans le document Seebeck.md
    """
    # @cython.cdivision(True)
    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    def __init__(self, arg={}):
        if arg != {}:
            self.arg.tp = arg["tp"]
            self.arg.tp2 = arg["tp2"]
            self.arg.T = arg["T"]
            self.arg.E = arg["E"]
            self.arg.Np = arg["Np"]
            self.arg.beta = 1.0 / self.arg.T
            self.arg.v = 2.0 * math.pi / float(self.arg.Np)

    #cpdef void initialisation(self, dict arg):
    def initialisation(self, arg):
            self.arg = <param>arg
            if self.arg.beta == 0.0:
                self.arg.beta = 1.0 / self.arg.T
            if self.arg.v == 0.0:
                self.arg.v = 2.0 * math.pi / float(self.arg.Np)

    cdef inline double eperp(self, long k):
        return -2.0 * self.arg.tp * math.cos(k * self.arg.v)\
               - 2.0 * self.arg.tp2 * math.cos(2.0 * k * self.arg.v)

    @cython.cdivision(True)
    cdef inline double sigma(self, double sum_eperp):
        cdef:
            double sig, sig_sur_T
            double sigma_value
            double Sinh

        sig = 0.25 * sum_eperp * self.arg.beta
        sig_sur_T = sig-0.5 * self.arg.E * self.arg.beta
        #version totale
        if math.fabs(sig)==0.0:
            sigma_value = 0.5/math.cosh(sig_sur_T)
        else:
            sigma_value = sig/math.sinh(2*sig)/math.cosh(sig_sur_T)
        if sigma_value*10**20 <= 1:
            sigma_value = 0.0

        # #autre version
        # sigma_value = 0.0
        # if (math.fabs(sig_sur_T) <= 100.0) and (math.fabs(sig) <= 100.0):
        #     if(math.fabs(sig) <= 10**-6):
        #         Sinh = 0.5
        #     else:
        #         Sinh = sig/math.sinh(2*sig)
        #     sigma_value = Sinh/math.cosh(sig_sur_T)

        return sigma_value # round(sigma_value,20)

    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    cpdef  double test_sigma(self, long[:] array):
        cdef int i
        cdef int N = array.shape[0]
        cdef double val =0.0
        for i in range(N):
            val += self.eperp(array[i])

        return self.sigma(val)

    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    cdef  double[:,:,:] sequentiel(self) :
        cdef:
            int N = self.arg.Np
            int i, j, k
            double v1, v2
            double s_v
            double[:,:,:] array
        array = np.zeros([N, N, N], dtype=np.double)


        for i in range(N):
            v1 = self.eperp(i)
            for j in range(i,N):
                v2= self.eperp(j)
                for k in range(N):
                    s_v = v1 + v2 + self.eperp(k)\
                            + self.eperp(i+j-k)
                    array[i, j, k] = self.sigma(s_v)
                    array[j, i, k] = array[i, j, k]
        return array

    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    cpdef double[:,:,:] calcul_matrice_mu(self):
        return self.sequentiel()
