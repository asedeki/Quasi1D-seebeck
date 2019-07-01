## cython: profile=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
import cython
cimport cython
import pathlib as plib
import math
import numpy as np
cimport numpy as np
import sys

cdef class System():
    def __init__(self, str g_file):
        self.g_file = g_file
        self._temperatures = np.zeros([0], dtype = np.double)
        self.parametres = self.get_parameters()
        self.N = self.parametres["Np"]

    property temperatures:
        def __get__(self):
            return np.array(self._temperatures)
        def __set__(self, Temps):
            self._temperatures = np.array(Temps, dtype = np.double)

    cdef dict get_parameters(self):
        cdef:
            dict _parameters_keys
            list lines, l
        _parameters_keys = {"N_patche":"Np", "t_perp":"tp",
                            "t_perp2":"tp2", "E_F":"ef", "Tau":"tau",
                            "g1_ini":"g1i", "g2_ini":"g2i","g3_ini":"g3i"}

        my_file = plib.Path(self.g_file)
        if not my_file.is_file():
            input("le fichier  % s inexistant" % self.g_file)
            sys.exit(1)

        with open(self.g_file, "r") as f:
            lines=[]
            while True:
                l = f.readline()[:-1].split("=")
                if l[0][0] != "#":
                    break
                if len(l)==2:
                    l[0] = l[0][2:]
                    lines.append(l)
        parametres = {}
        for ln in lines:
            if ln[0] in _parameters_keys.keys():
                try:
                    parametres[_parameters_keys[ln[0]]] = round(float(ln[1]), 3)
                except Exception as e:
                    print(f"{ln[0]} ne correspond pas a un parametre reconnu")
                    sys.exit(1)
            elif  ln[0][:-4] in _parameters_keys.keys():
                parametres[_parameters_keys[ln[0][:-4]]] = round(float(ln[1]), 3)

        parametres["Np"] = int(parametres["Np"])

        return parametres

    # def temperatures(self, double[:] temperatures):
    #     self.temperatures = temperatures

    cpdef set_interaction(self):
        """
        Permet de lire les valeurs des constantes de couplage
        g_1, g_2 ET g_3 a partir du fichier g_file.
        """
        cdef:
            np.ndarray[double, ndim=1] Temps
            long[:,:,:] array
            int N, ig
            str dirInd
            int lenT
            double Ti
            np.ndarray[double, ndim=1] G_i
            double[:] GT

        dirInd = "/Users/asedeki/MegaSynch/RecherchePhysique/"
        dirInd += "SherbrookeEte2017/inddata"  # My laptop
        N = self.N
        array = np.load(f"{dirInd}/array_index_n{N}.npy")

        Temps= np.loadtxt(self.g_file, usecols=[1], unpack=True, dtype=np.double)
        if self._temperatures.size == 0 :
            self.temperatures = np.unique(Temps)

        self.g = {}
        for i in [1,2,3]:
            self.g[i] = {}
            G_i = np.loadtxt(self.g_file, usecols=[4+i], unpack=True, dtype=np.double)
            for Ti in self.temperatures:
                GT = G_i[np.where(Temps == Ti)]
                self.g[i][Ti] = np.zeros([N,N,N], dtype = np.double)
                self.set_g(self.g[i][Ti], array, GT)

    cdef set_g(self,double[:,:,:] g, long[:,:,:] array, double[:] g_T):
        cdef:
            int N
            int i, j, k

        N = self.N
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    g[i,j,k] = g_T[array[i,j,k] - 1]
