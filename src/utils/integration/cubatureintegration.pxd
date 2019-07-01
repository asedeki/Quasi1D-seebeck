from Seebeck.src.utils.system.structure cimport param
cdef extern from './cfiles/cubature.h':
    # values for the error_norm parameter
    ctypedef enum error_norm:
        ERROR_INDIVIDUAL = 0
        ERROR_PAIRED
        ERROR_L2
        ERROR_L1
        ERROR_LINF

    ctypedef int (*integrand) (unsigned ndim, const double *x, void *fdata,
                               unsigned fdim, double *fval)

    ctypedef int (*integrand_v) (unsigned ndim, size_t npt, const double *x,
                                 void *fdata, unsigned fdim, double *fval)

    int hcubature(unsigned fdim, integrand f, void *fdata,
                  unsigned ndim, const double *xmin, const double *xmax,
                  unsigned maxEval, double reqAbsError, double reqRelError,
                  error_norm norm, double *val, double *err)

    int pcubature(unsigned fdim, integrand f, void *fdata,
                  unsigned ndim, const double *xmin, const double *xmax,
                  size_t maxEval, double reqAbsError, double reqRelError,
                  error_norm norm, double *val, double *err)

    int hcubature_v(unsigned fdim, integrand_v f, void *fdata,
                    unsigned ndim, const double *xmin, const double *xmax,
                    size_t maxEval, double reqAbsError, double reqRelError,
                    error_norm norm, double *val, double *err)

    int pcubature_v(unsigned fdim, integrand_v f, void *fdata,
                    unsigned ndim, const double *xmin, const double *xmax,
                    size_t maxEval, double reqAbsError, double reqRelError,
                    error_norm norm, double *val, double *err)
# Vectorized version with user-supplied buffer to store points and values.
# The buffer *buf should be of length *nbuf * dim on entry (these parameters
# are changed upon return to the final buffer and length that was used).
# The buffer length will be kept <= max(max_nbuf, 1) * dim.
#
# Also allows the caller to specify an array m[dim] of starting degrees
# for the rule, which upon return will hold the final degrees.  The
# number of points in each dimension i is 2^(m[i]+1) + 1.
    int pcubature_v_buf(unsigned fdim, integrand_v f, void *fdata,
                    unsigned ndim, const double *xmin, const double *xmax,
                    size_t maxEval, double reqAbsError, double reqRelError,
                    error_norm norm, unsigned *m,
                    double **buf, size_t *nbuf, size_t max_nbuf,
                    double *val, double *err)

ctypedef struct Pdata:
    double k1
    double k2
    param* P
    double ebeta


cdef double eperp(double kperp, double tp, double tp2)
cdef double sum_eperp(unsigned int nkperp, double* kperp, double tp, double tp2)
cdef double sigma(double sum_eperp, double ebeta)

cdef int diff_patch_3d(unsigned ndim, const double *x,
                void *fdata, unsigned fdim, double *fval)

cdef int diff_patch_2d_v(unsigned ndim, size_t npt, const double *x,
                        void *fdata, unsigned fdim, double *fval)

cdef int diff_patch_2d_mu2(unsigned ndim, size_t npt, const double *x,
                void *fdata, unsigned fdim, double *fval)

cdef int diff_patch_3d_v(unsigned ndim, size_t npt, const double *x,
                        void *fdata, unsigned fdim, double *fval)

# cdef int diff_patch_1d(unsigned ndim, const double *x,
#                 void *fdata, unsigned fdim, double *fval)

cdef int diff_patch_1d_v(unsigned ndim, size_t npt, const double *x,
                        void *fdata, unsigned fdim, double *fval)
cdef int diff_patch_1d2_v(unsigned ndim, size_t npt, const double *x,
                        void *fdata, unsigned fdim, double *fval)

############# New ##########################
ctypedef int (*integration_v)(integrand_v , void *, double*,double*, double*, double*, unsigned)
ctypedef int (*integration)(integrand , void *, double*,double*, double*, double*, unsigned)



cdef int integration_p_v(integrand_v , void *, double*,double*, double*, double*, unsigned)
cdef int integration_h_v(integrand_v , void *, 
            double*,double*, double*, double*, unsigned)
cdef int integration_h(integrand , void *, double*,double*, double*, double*, unsigned)
# cdef int integration_p(integrand , void *, double*,double*, double*, double*, unsigned)

# cdef int integration_Mu1(param , double*, str)
# cdef int integration_Mu2(param, double*, str)

cpdef get_mu_matrix(param parametres, double[:,:,::1], double[:,:,::1], str)
cdef int get_Mu1(param para, double[:,:,::1] Marray, str typ)
cdef int get_Mu2(param para, double[:,:,::1] Marray, str typ)
cdef int get_Mu_3d(param para, double[:,:,:,::1] Marray, str typ)
cdef int get_Mu_3d_v(param para, double[:,:,:,::1] Marray, str typ)
cdef int get_Mu2_4d(param para, double[:,:,:,::1] Marray, str typ)
# cdef int get_Mu2_h(param para, double[:,:,::1] Marray, str typ)
