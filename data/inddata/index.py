import numpy as np
from pprint import pprint
# file = "indexes_n16.dat"
# with open(file,"r") as f:
#     d = f.readlines()
# r =[x[:-1].strip(" ").split("  ") for x in d[1:]]
# r1 = [x for x in r if len(x)==4]
# r2 = [x for x in r if len(x)>4]
# for i in r2:
#     j = i.count("")
#     for _ in range(j):
#        i.remove("")
#     if len(i) !=4:
#         print(i)
#     r1.append(i)
# N = 16
# array = np.zeros([N, N, N], dtype=int)
# for (l,i,j,k) in r1:
#     array[(int(i)+N)%N, (int(j)+N)%N, (int(k)+N)%N] = int(l)
# pprint(array[N//2,N//2,N//2])
# np.save(f"array_index_n16", array)
#
#
# #dic = {(int(e[1]), int(e[2]), int(e[3])): int(e[0]) for e in r }
# #r = [w.strip() for w in m.split(" ") for m in d]
# N = 32
# inds = [(i, j,k) for i in range(N) for j in range(N) for k in range(N)]
#
# array = np.load("array_index_n32.npy")
# T, G = np.loadtxt("g.dat", usecols=[1,7], unpack=True, dtype=float)
# Temp = set(T)
# g3_array = {Ti: np.zeros([N, N, N], dtype=float) for Ti in Temp}
# for Ti in Temp:
#     g = G[np.where(T == Ti)]
#     for e in inds:
#         g3_array[Ti][e] = g[array[e]-1]
# pprint(g3_array[Ti])
a=[1]+[2]