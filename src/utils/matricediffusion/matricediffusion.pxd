# import pdb
import numpy as np
cimport numpy as np
#ctypedef struct param
from Seebeck.src.utils.system.structure cimport param


cdef class MatriceDiffusion:

    cdef readonly param arg
    cdef double[:,:,:] g3
    #cpdef void initialisation(self, dict arg,  )
    cdef inline double eperp(self, long k)
    cdef double sigma(self, double sum_eperp)
    cdef void get_sigma(self, double[:,:,::1], double[:,:,::1])
    cpdef double[:,:] get_collision_matrix(self)
    cdef double[:] get_row_collision_matrix(self,
                    double[:,:,::1] mu_1, double[:,:,::1] mu_2, long k1)
    cdef double get_ek_deriv(self, double e, double eta, double tp)
    cpdef double get_cst_Q0a(self, double eta, double tp)

cdef class MatriceDiffusionNew(MatriceDiffusion):
    cpdef double functionMu1(self, double, double, double)
    cpdef int get_Mu1(self, double[:,:,::1])
    cpdef double functionMu2(self, double, double, double)
    cpdef int get_Mu2(self, double[:,:,::1])
cdef class MatriceDiffusionInteg(MatriceDiffusion):

    cdef double[:] get_row_collision_matrix_intg(self,
                    double[:,:,::1] , double[:,:,:,::1], long)
    cdef void get_sigma_intg(self, double[:,:,::1], double[:,:,:,::1]) 