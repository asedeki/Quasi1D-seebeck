# import pdb
import numpy as np
cimport numpy as np
#ctypedef struct param
ctypedef struct param:
    double tp
    double tp2
    double E
    double T
    double beta
    unsigned int Np
    double v

ctypedef long long_t
cdef class MatriceMu:
    cdef readonly param arg
    #cpdef void initialisation(self, dict arg)
    cdef double eperp(self, long_t k)
    cdef double sigma(self, double sum_eperp)
    cdef double[:,:,:] sequentiel(self) 
    cpdef double test_sigma(self, long[:] array)
    cpdef double[:,:,:] calcul_matrice_mu(self)
