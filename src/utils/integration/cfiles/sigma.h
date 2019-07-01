
struct parametres{
    double tp;
    double tp2;
    double E;
    double T;
    double beta;
    long Np;
    double v;
};
typedef struct parametres param;
double eperp_c(double kperp, param* p);
double sigma_c(double sum_eperp , param* p);
void matrice_mu(long N, param* p, double *array);
