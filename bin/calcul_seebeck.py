#!/usr/local/bin/python3
from datetime import datetime
import posix
import posixpath
import time
import numpy as np
import os

from Seebeck.src.seebeck_c import Seebeck
import Seebeck.src.myoptik2 as myoptik
from Seebeck.src.myoptik2 import putFileinDir
from Seebeck.src.myoptik2 import scan
from Seebeck.src.utils.system.system import System
import concurrent.futures as concfut
import csv
import pathlib


def run_system():
    fileg = "g.dat"
    f1 = pathlib.Path("interactions.npy")
    if not f1.is_file():
        sys = System(fileg)
        sys.set_interaction()
        data = {
            "p": sys.parametres,
            "g": sys.g
        }
        np.save("interactions", data)
        return sys.parametres, sys.g[3]
    else:
        data = np.load("interactions.npy")
        parametres = data[()]["p"]
        #g1 = data[()]["g"][1]
        #g2 = data[()]["g"][2]
        g3 = data[()]["g"][3]
        return parametres, g3


def run():
    Ts = _O.temperatures
    output = _O.output
    energies = _O.energies
    parametres, g3 = run_system()
    if Ts is None:
        Temps = np.array(list(g3.keys()), dtype=np.double)
        Temps = Temps[Temps >= 1]
    else:
        Temps = Ts
    s = Seebeck(parametres=parametres, g3=g3,
                temperatures=Temps, energies=energies, integration=_O.integration)
    Qa = -s.coefficient_seebeck_Q0a()
    if _O.seebeck:
        #Qa = 0.02/np.pi**2
        seebeck_coef = []
        for t in Temps:
            try:
                s_cof, err = s.coefficient_seebeck(t)
                seebeck_coef.append((t, (s_cof+Qa)*t, err))
                #seebeck_coef.append((t,s_cof, err))
            except Exception as e:
                print(e)
        s.save_csv(output+"_Seebeck", keys=["Temperature",
                                            "CoefficientSeebeck", "Erreur"], data=seebeck_coef)
    if Ts is not None:
        TT = []
        for t in Ts:
            m = np.min(np.abs(Temps-t))
            TT.append(Temps[np.abs(Temps-t) == m][0])
        Temps = TT
    s.set_temps_diffusion(temperatures=Temps)
    s.save_csv(output+"_diffusion")


def run_dir(d):
    d = d.split("/")
    d = "/".join(d[:-1])
    print(d)
    posix.chdir(d)
    run()
    posix.chdir(_O.dirini)


def main(_O):
    dirini = os.getcwd()
    _O.dirini = dirini
    if _O.scan:
        scan_files = scan(".")  # scan_files = [] ;scan2(".", scan_files)
        if(len(scan_files) > 0):
            scan_files.sort()
            if _O.parallel:
                with concfut.ProcessPoolExecutor() as executor:
                    future1 = executor.map(run_dir, scan_files)
            else:
                [run_dir(d) for d in scan_files]
    else:
        t1 = time.time()
        run()

    # sc = np.load("scatering_time.npy")

    # with concfut.ProcessPoolExecutor() as executor:
    #      future = executor.map(q.get_scattering_time, [(T, q.energies[0]) for T in q.temperatures])
    # print(future)


if __name__ == '__main__':
    print(datetime.now().strftime('#$%Y-%m-%d %H:%M:%S'))
    _O = myoptik.opt()
    t1 = time.time()
    main(_O)
    print("#Temps Exec=%s" % round(time.time() - t1, 3))
    print(datetime.now().strftime('#$%Y-%m-%d %H:%M:%S'))
