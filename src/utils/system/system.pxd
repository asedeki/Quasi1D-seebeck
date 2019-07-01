cdef class System:
    cdef:
        public dict parametres
        public double[:] _temperatures
        str g_file
        readonly dict g

        int N
        cdef dict get_parameters(self)
        cpdef set_interaction(self)
        cdef set_g(self,double[:,:,:] g, long[:,:,:] array, double[:] g_T)
