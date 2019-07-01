#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <gsl/gsl_integration.h>

typedef struct{
    double tp;
    double tp2;
    double E;
    double T;
    double beta;
    long Np;
    double v;
}param;
double eperp_c(double kperp, param* p)
{
    double res=0.0 , k;
    k = kperp * p->v;
    res = -2.0 * p->tp * cos(k) - 2.0 * p->tp2 * cos(2.0 * k);
    return(res);
}

double sigma_c(double sum_eperp, param* p){
    double sig, sig_sur_T;
    double sigma_value;

    sig = 0.25 * sum_eperp * p->beta;
    sig_sur_T = sig-0.5 * p->E * p->beta;
    if (fabs(sig)==0.0){
        sigma_value = 0.5/cosh(sig_sur_T);
    } else{
        sigma_value = sig/sinh(2*sig)/cosh(sig_sur_T);
    }
    if(sigma_value*pow(10.0,20) <= 1.0){
        sigma_value = 0.0;
    }
    return sigma_value;
}
void matrice_mu(long N, param* p, double *array) {
    int l = 0;
    double v1, v2;
    double s_v;
    for(int i=0; i<N; i++){
        v1 = eperp_c(i, p);
        for(int j=0; j<N; j++){
            v2 = eperp_c(j, p);
            for(int k=0; k<N; k++){
                s_v = v1 + v2 + eperp_c(k, p)+ eperp_c(i+j-k, p);
                *(array+l) = sigma_c(s_v, p);
                l++;
            }
        }
    }
}
