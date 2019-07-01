from Seebeck.utils.matrice_mu_c cimport param, MatriceMu
from Seebeck.src.system import System

#from typing import Dict


cdef class Seebeck:
    cdef public double[:] temperatures
    cdef public double[:] energies
    cdef public param parametres
    cdef readonly MatriceMu MatMu
    cdef int inf, sup
    cdef dict g3
    cdef readonly dict seebeck
    cdef readonly dict scattering_time

    cdef double[:,:,:] get_mu(self, double t, double e)
    cdef double[:,:] get_collision_matrix(self, double T, double E)
    cdef double[:] get_row_collision_matrix(self, double[:,:,:] mu_2, long k1, double[:,:,:] g3)
