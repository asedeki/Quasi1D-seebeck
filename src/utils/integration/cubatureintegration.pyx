# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
## cython: profile=True
## distutils: extra_compile_args = -DVECTORISED

cimport numpy as np
import numpy as np
from libc cimport math
import cython
from cython.operator cimport dereference as deref
from libc.stdlib cimport malloc, free, calloc
import sys
import scipy
cdef double eperp(double kperp, double tp, double tp2):
    return  -2.0 * tp * math.cos(kperp) - 2.0 * tp2 * math.cos(2.0 * kperp)

cdef double sum_eperp(unsigned int nkperp, double* kperp, double tp, double tp2):
    cdef:
        double sum = 0.0
        unsigned int i
    for i in range(nkperp):
        sum += eperp(kperp[i], tp, tp2)
    return sum

cdef double sigma(double sig_beta, double e_beta):
    cdef:
        double sig_sur_T
        double sigma_value
        double inf1
        double inf2
    sig_sur_T = sig_beta -0.5 * e_beta
    sigma_value = 0.0
    inf1 = math.fabs(sig_beta)
    inf2 = math.fabs(sig_sur_T)
    if(inf1==0):
        sigma_value = 0.5/math.cosh(sig_sur_T)
    else:
        sigma_value = sig_beta/math.sinh(2*sig_beta)/math.cosh(sig_sur_T)
    # if not (inf1 >= 20 or inf2 >=20):
    #     if (inf1==0.0):
    #         sigma_value = 0.5/math.cosh(sig_sur_T)
    #     else:
    #         sigma_value = sig_beta/math.sinh(2*sig_beta)/math.cosh(sig_sur_T)
    return sigma_value

cdef int diff_patch_2d_mu2(unsigned ndim, size_t npt, const double *x,
                void *fdata, unsigned fdim, double *fval):
    cdef:
        unsigned int nkperp = 4
        double *kperp
        double s_eperp_beta = 0.0
        unsigned int i,j
        Pdata *Parametres = <Pdata*> fdata

    kperp = <double*>malloc(nkperp*sizeof(double))
    kperp[0] = Parametres.k1
    kperp[1] = Parametres.k2

    for j in range(npt):
        for i in range(ndim):
            kperp[2+i]= x[j*ndim+i]
        s_eperp_beta = sum_eperp(nkperp, kperp, Parametres.P.tp, Parametres.P.tp2)
        s_eperp_beta *= 0.25 * Parametres.P.beta
        fval[j]= sigma(s_eperp_beta, Parametres.ebeta)
    return 0

cdef int diff_patch_2d_v(unsigned ndim, size_t npt, const double *x,
                void *fdata, unsigned fdim, double *fval):
    cdef:
        unsigned int nkperp = 4
        double *kperp
        double s_eperp_beta = 0.0
        unsigned int i,j
        Pdata *Parametres = <Pdata*> fdata

    kperp = <double*>malloc(nkperp*sizeof(double))
    kperp[0] = Parametres.k1

    for j in range(npt):
        kperp[1] = -kperp[0]
        for i in range(ndim):
            kperp[2+i]= x[j*ndim+i]
            kperp[1] += kperp[2+i]
        s_eperp_beta = sum_eperp(nkperp, kperp, Parametres.P.tp, Parametres.P.tp2)
        s_eperp_beta *= 0.25 * Parametres.P.beta
        fval[j]= sigma(s_eperp_beta, Parametres.ebeta)
    return 0

cdef int diff_patch_3d_v(unsigned ndim, size_t npt, const double *x,
                void *fdata, unsigned fdim, double *fval):
    cdef:
        unsigned int nkperp = 4
        double *kperp
        double s_eperp_beta = 0.0
        unsigned int i,j
        double sumf=0.0
        Pdata *Parametres = <Pdata*> fdata

    kperp = <double*>malloc(nkperp*sizeof(double))
    kperp[0] = Parametres.k1
    for j in range(npt):
        for i in range(ndim):
            kperp[i+1]= x[j*ndim+i]
        s_eperp_beta = sum_eperp(nkperp, kperp, Parametres.P.tp, Parametres.P.tp2)
        s_eperp_beta *= 0.25 * Parametres.P.beta
        fval[j]= sigma(s_eperp_beta, Parametres.ebeta)
        #sumf += fval[j]
    #print(sumf/npt)
    return 0

cdef int diff_patch_3d(unsigned ndim, const double *x,
                void *fdata, unsigned fdim, double *fval):
    cdef:
        unsigned int nkperp = 4
        double *kperp
        double s_eperp_beta = 0.0
        unsigned int i
        #double sumf=0.0
        Pdata *Parametres = <Pdata*> fdata

    kperp = <double*>malloc(nkperp*sizeof(double))
    kperp[0] = Parametres.k1
    for i in range(ndim):
        kperp[i+1]= x[i]
    s_eperp_beta = sum_eperp(nkperp, kperp, Parametres.P.tp, Parametres.P.tp2)
    s_eperp_beta *= 0.25 * Parametres.P.beta
    fval[0] = 0.0
    if math.fabs(s_eperp_beta) <= 100.0:
        fval[0]= sigma(s_eperp_beta, Parametres.ebeta)
    #print(ndim, kperp[0], x[0],fval[0])
    return 0

cdef int diff_patch_1d_v(unsigned ndim, size_t npt, const double *x,
                void *fdata, unsigned fdim, double *fval):
    cdef:
        unsigned int nkperp = 4
        double *kperp
        double s_eperp_beta
        unsigned int i,j
        Pdata *Parametres = <Pdata*> fdata

    kperp = <double*>malloc(nkperp*sizeof(double))
    kperp[0] = Parametres.k1
    kperp[1] = Parametres.k2

    for j in range(npt):
        kperp[2] = x[j]
        kperp[3] = kperp[0] + kperp[1] - kperp[2]
        s_eperp_beta = sum_eperp(nkperp, kperp, Parametres.P.tp, Parametres.P.tp2)
        s_eperp_beta *= 0.25 * Parametres.P.beta
        fval[j]= sigma(s_eperp_beta, Parametres.ebeta)
    return 0

cdef int diff_patch_1d2_v(unsigned ndim, size_t npt, const double *x,
                void *fdata, unsigned fdim, double *fval):
    cdef:
        unsigned int nkperp = 4
        double *kperp
        double s_eperp_beta
        unsigned int i,j
        Pdata *Parametres = <Pdata*> fdata

    kperp = <double*>malloc(nkperp*sizeof(double))
    kperp[0] = Parametres.k1
    kperp[1] = Parametres.k2

    for j in range(npt):
        kperp[2] = x[j]
        kperp[3] = kperp[0] + kperp[2] - kperp[1]
        s_eperp_beta = sum_eperp(nkperp, kperp, Parametres.P.tp, Parametres.P.tp2)
        s_eperp_beta *= 0.25 * Parametres.P.beta
        fval[j]= sigma(s_eperp_beta, Parametres.ebeta)
    return 0

cdef int integration_h_v(integrand_v fg, void *fdata, double* val,
                    double* err, double* xmin, double* xmax, unsigned xdim):
    cdef:
        unsigned fdim=1
        double reqRelError=1e-4
        size_t maxEval = 0
        double reqAbsError = 1e-4
    hcubature_v(fdim, fg, fdata, xdim, xmin, xmax, maxEval,
                reqAbsError, reqRelError, error_norm.ERROR_INDIVIDUAL, val, err)
    return 0

cdef int integration_h(integrand fg, void *fdata, double* val,
                    double* err, double* xmin, double* xmax, unsigned xdim):
    cdef:
        unsigned fdim=1
        double reqRelError=1e-3
        size_t maxEval = 0
        double reqAbsError = 0
    hcubature(fdim, fg, fdata, xdim, xmin, xmax, maxEval,
                reqAbsError, reqRelError, error_norm.ERROR_INDIVIDUAL, val, err)
    return 0

cdef int integration_p_v(integrand_v fg, void *fdata, double* val,
                            double* err, double* xmin, double* xmax, unsigned xdim):
        cdef:
            unsigned fdim=1
            double reqRelError=1e-3
            size_t maxEval = 0
            double reqAbsError = 0
        pcubature_v(fdim, fg, fdata, xdim, xmin, xmax, maxEval,
                    reqAbsError, reqRelError, error_norm.ERROR_INDIVIDUAL, val, err)
        return 0

cpdef get_mu_matrix(param parametres, double[:,:,::1] Mu1, double[:,:,::1] Mu2, str typ):
    cdef:
        int value

    value = get_Mu1(parametres, Mu1, typ)
    value = get_Mu2(parametres, Mu2, typ)


cdef int get_Mu1(param para, double[:,:,::1] Marray, str typ):
    cdef:
        unsigned int k1,k3,k4
        unsigned int N = para.Np
        void* fdata
        double xmin[2]
        double xmax[2]
        unsigned xdim = 2
        double delta
        double val, err
        integrand_v funct_int
        Pdata pdata
        double epsilon[2]
        double d_epsilon
        integration_v f_integration

    if typ == "h" :
        f_integration = integration_h_v
    elif typ == "p" :
        f_integration = integration_p_v
    else:
        print("Usage 'typ= h' ou 'typ = p'")
        sys.exit()

    pdata.P = &para
    pdata.ebeta = para.beta * para.E
    delta = para.v
    fdata = &pdata
    epsilon[0] = -delta*0.5
    epsilon[1] = delta*0.5
    # epsilon[1] = 0.0
    # epsilon[0] = -delta
    d_epsilon = epsilon[1] - epsilon[0]     
    for k1 in range(N):
        pdata.k1 = k1*delta
        pdata.k2 = k1*delta
        for k3 in range(N):
            xmin[0] = k3*delta + epsilon[0]
            xmax[0] = k3*delta + epsilon[1]
            for k4 in range(k3,N):
                xmin[1] = k4*delta + epsilon[0]
                xmax[1] = k4*delta + epsilon[1]
                f_integration(diff_patch_2d_v, fdata, &val, &err, xmin,xmax,xdim)
                # if abs(val) <= 1e-20:
                #     val = 0.0
                Marray[k1][k3][k4] = val/(d_epsilon**2)
                Marray[k1][k4][k3] = val/(d_epsilon**2)

    return 0

cdef int get_Mu2(param para, double[:,:,::1] Marray, str typ):
        cdef:
            unsigned int k1,k2,k3
            unsigned int N = para.Np
            void* fdata
            double xmin[1]
            double xmax[1]
            unsigned xdim = 1
            double delta
            double val, err
            double epsilon[2]
            Pdata pdata
            integrand_v f_integration
            double d_epsilon

        if typ == "0" :
            f_integration = diff_patch_1d_v
        elif typ == "1" :
            f_integration = diff_patch_1d2_v
        else:
            print("Usage 'typ= h' ou 'typ = p'")
            sys.exit()

        pdata.P = &para
        pdata.ebeta = para.beta * para.E
        delta = para.v
        fdata = &pdata
        epsilon[0] = -delta*0.5
        epsilon[1] = delta*0.5
        
        d_epsilon = epsilon[1] - epsilon[0]    
        for k3 in range(N):
            xmin[0] = k3*delta + epsilon[0]
            xmax[0] = k3*delta + epsilon[1]
            #print(k3*delta, xmin, xmax)
            for k1 in range(N):
                pdata.k1 = k1*delta
                for k2 in range(k1,N):
                    pdata.k2 = k2*delta            
                    integration_h_v(f_integration, fdata, &val, &err, 
                                    xmin,xmax, xdim)
                    # if abs(val) <= 1e-20:
                    #     val = 0.0
                    Marray[k1][k2][k3] = val/d_epsilon
                    Marray[k2][k1][k3] = Marray[k1][k2][k3]

        return 0

cdef int get_Mu_3d(param para, double[:,:,:,::1] Marray, str typ):
    cdef:
        unsigned int k1, k2, k3, k4
        unsigned int N = para.Np
        void* fdata
        double xmin[3]
        double xmax[3]
        unsigned xdim = 3
        double delta
        double val, err
        integrand_v funct_int
        Pdata pdata
        double epsilon[2]
        double d_epsilon
        integration f_integration

    if typ == "h" :
        f_integration = integration_h
    elif typ == "p" :
        f_integration = integration_h
    else:
        print("Usage 'typ= h' ou 'typ = p'")
        sys.exit()

    pdata.P = &para
    pdata.ebeta = para.beta * para.E
    delta = para.v
    fdata = &pdata
    epsilon[0] = -delta*0.5
    epsilon[1] = delta*0.5
    # epsilon[1] = 0.0
    # epsilon[0] = -delta
    d_epsilon = epsilon[1] - epsilon[0]     
    for k1 in range(N):
        pdata.k1 = k1*delta
        for k2 in range(N):
            xmin[0] = k2*delta - epsilon[0]
            xmax[0] = k2*delta + epsilon[1]
            for k3 in range(k2, N):
                xmin[1] = k3*delta - epsilon[0]
                xmax[1] = k3*delta + epsilon[1]
                for k4 in range(k3, N):
                    xmin[2] = k4*delta - epsilon[0]
                    xmax[2] = k4*delta + epsilon[1]
                    #if (k1 == (k3+k4-k2)%N) or (k1 == (k4+k2-k3)%N):
                    f_integration(diff_patch_3d, fdata, 
                                &val, &err, xmin,xmax,xdim)
                    Marray[k1][k2][k3][k4] = val/(d_epsilon**3)
    return 0

cdef int get_Mu_3d_v(param para, double[:,:,:,::1] Marray, str typ):
    cdef:
        unsigned int k1, k2, k3, k4
        unsigned int N = para.Np
        void* fdata
        double xmin[3]
        double xmax[3]
        unsigned xdim = 3
        double delta
        double val, err
        integrand_v funct_int
        Pdata pdata
        double epsilon[2]
        double d_epsilon
        integration_v f_integration

    if typ == "h" :
        f_integration = integration_h_v
    elif typ == "p" :
        f_integration = integration_p_v
    else:
        print("Usage 'typ= h' ou 'typ = p'")
        sys.exit()

    pdata.P = &para
    pdata.ebeta = para.beta * para.E
    delta = para.v
    fdata = &pdata
    epsilon[0] = -delta*0.5
    epsilon[1] = delta*0.5
    # epsilon[1] = 0.0
    # epsilon[0] = -delta
    d_epsilon = epsilon[1] - epsilon[0]     
    for k1 in range(N):
        print(k1)
        pdata.k1 = k1*delta
        for k2 in range(N):
            xmin[0] = k2*delta + epsilon[0]
            xmax[0] = k2*delta + epsilon[1]
            for k3 in range(N):
                xmin[1] = k3*delta + epsilon[0]
                xmax[1] = k3*delta + epsilon[1]
                for k4 in range(N):
                    val = 0.0
                    xmin[2] = k4*delta + epsilon[0]
                    xmax[2] = k4*delta + epsilon[1]
                    f_integration(diff_patch_3d_v, fdata, &val, 
                                    &err, xmin,xmax,xdim)
                    Marray[k1][k2][k3][k4] = val/(d_epsilon**3)
    return 0

cdef int get_Mu2_4d(param para, double[:,:,:,::1] Marray, str typ):
    cdef:
        unsigned int k1, k2, k3, k4
        unsigned int N = para.Np
        void* fdata
        double xmin[2]
        double xmax[2]
        unsigned xdim = 2
        double delta
        double val, err
        integrand_v funct_int
        Pdata pdata
        double epsilon[2]
        double d_epsilon
        integration_v f_integration

    if typ == "h" :
        f_integration = integration_h_v
    elif typ == "p" :
        f_integration = integration_p_v
    else:
        print("Usage 'typ= h' ou 'typ = p'")
        sys.exit()

    pdata.P = &para
    pdata.ebeta = para.beta * para.E
    delta = para.v
    fdata = &pdata
    epsilon[0] = -delta*0.5
    epsilon[1] = delta*0.5
    # epsilon[1] = 0.0
    # epsilon[0] = -delta
    d_epsilon = epsilon[1] - epsilon[0]     
    for k1 in range(N):
        pdata.k1 = k1*delta
        for k2 in range(k1,N):
            pdata.k2 = k2*delta
            for k3 in range(N):
                xmin[0] = k3*delta + epsilon[0]
                xmax[0] = k3*delta + epsilon[1]
                for k4 in range(k3, N):
                    val = 0.0
                    xmin[1] = k4*delta + epsilon[0]
                    xmax[1] = k4*delta + epsilon[1]
                    f_integration(diff_patch_2d_mu2, fdata, &val, 
                                    &err, xmin,xmax,xdim)
                    Marray[k1][k2][k3][k4] = val/(d_epsilon**2)
                    Marray[k1][k2][k4][k3] = Marray[k1][k2][k3][k4]
                    Marray[k2][k1][k3][k4] = Marray[k1][k2][k3][k4]
                    Marray[k2][k1][k4][k3] = Marray[k1][k2][k3][k4]
    return 0