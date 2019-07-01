## cython: boundscheck=False
## cython: wraparound=False
# cython: cdivision=True
## cython: nonecheck=False

## cython: profile=True

cimport cython
import numpy as np
cimport numpy as np
from libc cimport math

# cimport Seebeck.utils.cubatureintegration as cbi
# import  Seebeck.utils.cubatureintegration as cbi
cimport Seebeck.utils.integration.cubatureintegration as cbi
import  Seebeck.utils.integration.cubatureintegration as cbi
from scipy.integrate import dblquad, quad, tplquad


cdef class MatriceDiffusion:
    """
        Calcul de la matrice Mu dans le document Seebeck.md
    """
    def __init__(self, arg={}, g3 = None):
        if arg != {}:
            self.arg.tp = arg["tp"]
            self.arg.tp2 = arg["tp2"]
            self.arg.T = arg["T"]
            self.arg.E = arg["E"]
            self.arg.Np = arg["Np"]
            self.arg.beta = 1.0 / self.arg.T
            self.arg.v = 2.0 * math.pi / float(self.arg.Np)
        if g3 is not None:
            self.g3 = g3

    def initialisation(self, param arg, double[:,:,:] g3):
        self.arg = <param>arg
        if self.arg.beta == 0.0:
            self.arg.beta = 1.0 / self.arg.T
        if self.arg.v == 0.0:
            self.arg.v = 2.0 * math.pi / float(self.arg.Np)
        self.g3 = g3

    cdef inline double eperp(self, long k):
        cdef:
            double kperp = k*self.arg.v
        return cbi.eperp(kperp, self.arg.tp, self.arg.tp2)

    cdef double sigma(self, double sum_eperp):
        cdef:
            double sig = 0.25 * self.arg.beta * sum_eperp
            double ebeta = self.arg.E * self.arg.beta
            double sigma_value
        sigma_value = cbi.sigma(sig, ebeta)
        return sigma_value

    cdef void get_sigma(self, double[:,:,::1] mu_1, double[:,:,::1] mu_2) :
        cdef:
            int N = self.arg.Np
            int i, j, k
            double v1, v2
            double s_v
            double val

        for i in range(N):
            v1 = self.eperp(i)
            for j in range(i,N):
                v2= self.eperp(j)
                for k in range(N):
                    s_v = v1 + v2 + self.eperp(k)\
                              + self.eperp(i+j-k)
                    val =  self.sigma(s_v)
                    if abs(val) <= 1e-20:
                        val = 0.0
                    mu_2[i][j][k] = val
                    mu_2[j][i][k] = mu_2[i][j][k]

        for i in range(N):
            for j in range(N):
                for k in range(N):
                    mu_1[i][j][k] = mu_2[k][j][i]
    
    
    cpdef double[:,:] get_collision_matrix(self):
        cdef:
            long k1
            int N = self.arg.Np
            double[:,:] collision_matrix
            double[:,:,::1] mu_2, mu_1
        mu_2 = np.zeros([N, N, N], dtype=np.double)
        mu_1 = np.zeros([N, N, N], dtype=np.double)
        self.get_sigma(mu_1, mu_2)

        collision_matrix = np.empty([N, N], dtype=np.double)
        for k1 in range(N) :
            collision_matrix[k1] = self.get_row_collision_matrix(mu_1, mu_2, k1)

        collision_matrix = np.array(collision_matrix) + np.array(collision_matrix.T) - \
            np.diag(np.diag(collision_matrix))

        return collision_matrix


    cdef double[:] get_row_collision_matrix(self, double[:,:,::1] mu_1,
                                            double[:,:,::1] mu_2, long k1) :
        cdef:
            #np.ndarray[double, ndim=1] row_col_matrix
            double[:] row_col_matrix
            long k2, k3, k4, i
            int N = self.arg.Np
            double g3_1=0.0, g3_2=0.0, g3_3=0.0, S2_1=0.0, S2_2=0.0
            double[:,:,:] g3
        g3 = self.g3
        row_col_matrix = np.zeros([N],dtype=np.double)
        row_col_matrix[k1] = 0.0
        for k3 in range(N):  # self.inf,self.sup
            for k4 in range(N):
                g3_1=0.0
                i = (k3 + k4 - k1)%N
                g3_1 = abs(g3[k1, i, k3] - g3[k1, i, k4]) ** 2
                row_col_matrix[k1] += mu_1[k1,k3, k4] * g3_1

        for k2 in range(k1+1,N):
            #if k2 != k1:
            row_col_matrix[k2] = 0.0
            for k3 in range(N):
                g3_2=0.0;g3_3=0.0
                S2_1=0.0; S2_2=0.0
                k4 = (k1 + k2 - k3)%N
                S2_1 = mu_2[k1, k2, k3]
                g3_2 = abs(g3[(k1, k2, k3)] - g3[(k1, k2, k4)]) ** 2
                k4 = (k1 + k3 - k2)%N
                S2_2 = mu_2[k1, k3, k2]
                g3_3 = abs(g3[(k1, k3, k2)] - g3[(k1, k3, k4)]) ** 2
                row_col_matrix[k2] += (g3_2 * S2_1 - 2 * g3_3 * S2_2)
        
        return row_col_matrix    

    cdef double get_ek_deriv(self, double e, double eta, double tp):
        cdef:
            int i, N = self.arg.Np
            double e_p
            double etap = 1+eta**2
            double etam = 1-eta**2
            double tr = 1.0/tp/math.sqrt(2.0)
            double Va, Na,ve
            double Ef = 3000.0#0.5*math.pi*tp/math.sqrt(2.0)
        Va = 0.0
        Na = 0.0
        for i in range(N):
            e_p = e - self.eperp(i)+Ef
            ve = math.sqrt(etam -((e_p*tr)**2-etap)**2)/e_p
            Va += ve
            Na += 1.0/ve

        return Va**2*Na

    cpdef double get_cst_Q0a(self, double eta, double tp):
        cdef:
            double e = 0.001
            double Q
            double[:] ve, cst
            int i
        ve = np.array([-3*e, -2*e, -e, e, 2*e, 3*e])
        cst = np.array([-1.0/60.0, 3.0/20.0, -3.0/4.0 , 3.0/4.0,-3/20.0, 1.0/60.0])
        Q = 0.0
        for i in range(ve.size):
            Q += cst[i]*math.log(self.get_ek_deriv(ve[i], eta, tp))
        Q /= e
        return Q

        
cdef class MatriceDiffusionNew(MatriceDiffusion):
    def __init__(self, arg={}, g3 = None):
        super().__init__(arg, g3)
    
    cpdef double functionMu1(self, double x, double y, double k1):
        cdef:
            double sum_e
        sum_e = cbi.eperp(k1, self.arg.tp, self.arg.tp2)
        sum_e += cbi.eperp(-k1+x+y, self.arg.tp, self.arg.tp2)
        sum_e += cbi.eperp(x, self.arg.tp, self.arg.tp2)
        sum_e += cbi.eperp(y, self.arg.tp, self.arg.tp2)
        
        return self.sigma(sum_e)

    cpdef double functionMu2(self, double x, double k1, double k2):
        cdef:
            double sum_e
        sum_e = cbi.eperp(k1, self.arg.tp, self.arg.tp2)
        sum_e += cbi.eperp(k2, self.arg.tp, self.arg.tp2)
        sum_e += cbi.eperp(x, self.arg.tp, self.arg.tp2)
        sum_e += cbi.eperp(k1+k2-x, self.arg.tp, self.arg.tp2)
        
        return self.sigma(sum_e)

    cpdef int get_Mu1(self, double[:,:,::1] mu_1):
        cdef:
            unsigned int k1, k3, k4
            unsigned int N = self.arg.Np
            double delta = self.arg.v
            double val, x1, x2, y1, y2
            double kk1
        for k1 in range(N):
            kk1 = k1*delta
            for k3 in range(N):
                x1 = (k3-0.5)*delta
                x2 = (k3+0.5)*delta
                for k4 in range(k3, N):
                    y1 = (k4-0.5)*delta
                    y2 = (k4+0.5)*delta
                    val = dblquad( self.functionMu1, x1, x2, y1, y2,
                        args=(kk1, ), epsrel=1.0e-03)[0]
                    mu_1[k1][k3][k4] = val/delta**2
            
        for k1 in range(N):
            for k3 in range(N):
                for k4 in range(0, k3):
                    mu_1[k1][k3][k4] = mu_1[k1][k4][k3]

        return 0
    
    cpdef int get_Mu2(self, double[:,:,::1] mu_2):
        cdef:
            unsigned int k1, k2, k3
            unsigned int N = self.arg.Np
            double delta = self.arg.v
            double val, x1, x2
            double kk1, kk2
        for k1 in range(N):
            kk1 = k1*delta
            for k2 in range(k1, N):
                kk2 = k2*delta
                for k3 in range(N):
                    x1 = (k3-0.5)*delta
                    x2 = (k3+0.5)*delta
                    val = quad( self.functionMu2, x1, x2,
                        args=(kk1, kk2, ), epsrel=1.0e-03)[0]
                    mu_2[k1][k2][k3] = val/delta
            
        for k1 in range(N):
            for k2 in range(0,k1):
                for k3 in range(N):
                    mu_2[k1][k2][k3] = mu_2[k2][k1][k3]

        return 0
    
    

    cdef void get_sigma(self, double[:,:,::1] mu_1, double[:,:,::1] mu_2) :
        cdef:
            int value
            str typ = "h"
            unsigned int N = self.arg.Np
            unsigned i, j, k
            double[:,:,:,::1] Mu
        
        value = self.get_Mu1(mu_1)
        value = self.get_Mu2(mu_2)

cdef class MatriceDiffusionInteg(MatriceDiffusion):
    
    def __init__(self, arg={}, g3 = None):
        super().__init__(arg, g3)
    
    cpdef double[:,:] get_collision_matrix(self):
        cdef:
            long k1
            int N = self.arg.Np
            double[:,:] collision_matrix
            double[:,:,::1] mu_1
            double[:,:,:,::1] mu_2
        mu_2 = np.zeros([N,N, N, N], dtype=np.double)
        mu_1 = np.zeros([N, N, N], dtype=np.double)
        self.get_sigma_intg(mu_1, mu_2)

        collision_matrix = np.empty([N, N], dtype=np.double)
        for k1 in range(N) :
            collision_matrix[k1] = self.get_row_collision_matrix_intg(
                                                        mu_1, mu_2, k1)
        collision_matrix = np.array(collision_matrix) +\
            np.array(collision_matrix.T) - np.diag(np.diag(collision_matrix))

        return collision_matrix


    cdef double[:] get_row_collision_matrix_intg(self,
                    double[:,:,::1] mu_1, double[:,:,:,::1] mu_2, long k1):
        cdef:
            #np.ndarray[double, ndim=1] row_col_matrix
            double[:] row_col_matrix
            long k2, k3, k4, i
            int N = self.arg.Np
            double g3_1=0.0, g3_2=0.0, g3_3=0.0, S2_1=0.0, S2_2=0.0
            double[:,:,:] g3

        g3 = self.g3
        row_col_matrix = np.zeros([N],dtype=np.double)
        row_col_matrix[k1] = 0.0
        for k3 in range(N):  # self.inf,self.sup
            for k4 in range(N):
                g3_1=0.0
                i = (k3 + k4 - k1)%N
                if (0<= i <N):
                    g3_1 = abs(g3[k1, i, k3] - g3[k1, i, k4]) ** 2
                row_col_matrix[k1] += mu_1[k1,k3, k4] * g3_1

        for k2 in range(k1+1,N):
            #if k2 != k1:
            row_col_matrix[k2] = 0.0
            for k3 in range(N):
                g3_2=0.0;g3_3=0.0
                S2_1=0.0; S2_2=0.0
                k4 = (k1 + k2 - k3)%N
                if (0<= k4<N):
                    S2_1 = mu_2[0, k1, k2,k3]
                    g3_2 = abs(g3[(k1, k2, k3)] - g3[(k1, k2, k4)]) ** 2
                k4 = (k1 + k3 - k2)%N
                if (0<= k4<N):
                    S2_2 = mu_2[1,k1, k2,k3]
                    g3_3 = abs(g3[(k1, k3, k2)] - g3[(k1, k3, k4)]) ** 2
                row_col_matrix[k2] += (g3_2 * S2_1 - 2 * g3_3 * S2_2)
        
        return row_col_matrix

    cdef void get_sigma_intg(self, double[:,:,::1] mu_1, double[:,:,:,::1] mu_2) :
        cdef:
            int value
            str typ_mu1 = "h"
            unsigned int N = self.arg.Np
            unsigned i, j, k
            double[:,:,:,::1] Mu
        print("In integration sum")

        value = cbi.get_Mu1(self.arg, mu_1, typ_mu1)
        value = cbi.get_Mu2(self.arg, mu_2[0], "0")
        value = cbi.get_Mu2(self.arg, mu_2[1], "1")
        #value = cbi.get_Mu2_4d(self.arg, mu_2, "p")
        # MM = np.array(Mu)
        # Mu = np.zeros([N,N,N,N], dtype=np.double)
        # value = cbi.get_Mu_3d_v(self.arg, Mu, "h")
        # for i in range(N):
        #     for j in range(N):
        #         for k in range(N):
        #             mu_1[i][j][k] = Mu[i,(j+k-i)%N,j,k] 
        #             #mu_2[i][j][k] = Mu[i,j,k,(i+k-j)%N]
    
