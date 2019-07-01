#!/usr/local/bin/python3
import sys
print(sys.argv)
input()
name = sys.argv[1].split(".")[0]+".gnu"
with open(sys.argv[1],"r") as f:
    lines=f.readlines()
for i in range(len(lines)):
    lines[i] = lines[i].replace(",","\t")

with open(name,"w") as f :
    f.writelines(lines[1:])
