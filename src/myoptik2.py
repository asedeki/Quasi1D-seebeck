import numpy as num
import sys
from optparse import OptionParser
import os
def to_array(v):
    if v[-1]=="l":
        v=[float(i) for i in v[:-1]]
        #a = num.linspace(v[0], v[1], int(v[2])).round(2)
        # x = (v[1]-v[0])/v[2]
        # a = num.arange(v[0],v[1]+x,x).round(2)
        a = num.arange(v[0],v[1]*1.01,v[2]).round(2)
    else:
        a = num.array(v, dtype=float)
    return a
def read_value(value, varname, vartype):
    """
    :rtype: object
    """
    v_tmp = value.split(",")
    if vartype == "float":
        v_tmp =value.split(":")
        v=num.array([])

        try:
            for v1 in v_tmp:
                v2 = to_array(v1.split(","))
                v = num.concatenate((v2,v))
            v = list(v)
            v.sort()
            v = num.array(v)
        except Exception as e:
            print("""
                    %s -> usage:E_1,E_N,N,l
                    ou
                    E_1,E_N,N,l:EE_1,EE_N,N,l:...
                    ou
                    E
                    """%varname)
            print(e)
            sys.exit(1)

    elif vartype == "string":
        v = value.split(",")
        if len(v) == 1:
            v = [value]

    return v

def Carargs(option, value, parser):
    assert value is None
    value = []
    rargs = parser.rargs
    while rargs:
        arg = rargs[0]
        if ((arg[:2] == "--" and len(arg) > 2) or
                (arg[:1] == "-" and len(arg) > 1 and arg[1] != "-")):
                break
        else:
            value.append((arg))
            del rargs[0]
    setattr(parser.values, option.dest, value)


def opt():
    parser = OptionParser()
    parser.add_option("-g", action="store", dest="g_file", type="string",
                      default="g.dat", help="""le nom du fichier contenant les valeurs
                                          des interactions: g_3
                                       """)

    parser.add_option("-o", "--output" ,action="store", dest="output", type="string",
                      default="tmp", help="""le nom du fichier de sortie
                                       """)

    parser.add_option('-e', action="callback",default=num.linspace(-20,20,50),dest="energies",
                      callback=callback_gen, type="string", nargs=1,
                      callback_args=('energies', 'float'),
                      help=" valeur min,max,nombre_pts de vecteur energie")

    parser.add_option('-t', action="callback",default=None,dest="temperatures",
                      callback=callback_gen, type="string", nargs=1,
                      callback_args=('temperatures', 'float'),
                      help=" valeur min,max,nombre_pts de temperatures")

    parser.add_option("--scan", action="store_true",
                      dest="scan", help="Imprime la matrice L")

    parser.add_option("--parallel", action="store_true",
                      dest="parallel", help="calcul parallel")

    parser.add_option("--integral", action="store_true",
                      dest="integral", help="calcul parallel")

    parser.add_option("--integration","-i",action="store_true",
                      dest="integration", help="Utilise la version Integration Pathches")

    parser.add_option("-s", "--seebeck", action="store_true",
                      dest="seebeck", help="calcul Coeff Seebeck")

    (_val, args) = parser.parse_args()
    return _val

def setup_opt():
    parser = OptionParser()
    parser.add_option("-m", action="store", dest="module", type="string",
                      default="", help="le nom du module")

    parser.add_option('-c', action="callback",default=[],dest="cfiles",
                      callback=callback_gen, type="string", nargs=1,
                      callback_args=('cfiles', 'string'), help="les fichiers c")

    parser.add_option('-p', "--pyx", action="callback",default=[],dest="pfiles",
                      callback=callback_gen, type="string", nargs=1,
                      callback_args=('pfiles', 'string'),
                      help=" les fichiers cython '.pyx' ")

    parser.add_option("-e", action="store_true",
                      dest="build", help="executer python setup.py ....")

    (_val, args) = parser.parse_args()
    return _val

def callback_gen(option, opt_here, value, parser, varname, vartype):
    v = read_value(value, varname, vartype)
    setattr(parser.values, varname, v)

######################## Autre chise #######################



def putFileinDir(dir, f):
    import filecmp
    test = False
    for f1 in dir:
        test = (test or filecmp.cmp(f1, f))
    if(not test):
        dir.append(f)


def scan(dir, part_file="*g.dat"):
    from pathlib import Path
    import pathlib
    dirini = os.getcwd()
    #path = Path(dir)
    path = pathlib.Path(dir)
    files = path.rglob('%s' % part_file)
    f = str(next(files)).split(part_file)[0]
    diff_files = [f"{dirini}/{f}"]
    while True:
        try:
            f = str(next(files)).split(part_file)[0]
            f = f"{dirini}/{f}"
            putFileinDir(diff_files, f)
        except StopIteration:
            break
    return diff_files

def scan2(dir, part_file="*.csv"):
    from pathlib import Path
    import pathlib
    dirini = os.getcwd()
    path = pathlib.Path(dir)
    files = path.rglob('%s' % part_file)
    print(files)
    return [str(f) for f in files]
    # for f in files:
    # f = str(next(files)).split(part_file)[0]
    # diff_files = [f"{dirini}/{f}"]
    # while True:
    #     try:
    #         f = str(next(files)).split(part_file)[0]
    #         f = f"{dirini}/{f}"
    #         putFileinDir(diff_files, f)
    #     except StopIteration:
    #         break
    # return diff_files
