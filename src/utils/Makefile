all: system.so cubatureintegration.so matricediffusion.so

system.so:
	cd ./system; make

cubatureintegration.so :
	cd ./integration; make

matricediffusion.so: matricediffusion/matricediffusion.pyx
		cd ./matricediffusion/;make

clean:
	cd ./system; make clean
	cd ./integration; make clean
	cd ./matricediffusion/;make clean

cleanall:
	cd ./system; make cleanall
	cd ./integration; make cleanall
	cd ./matricediffusion/;make cleanall

install:
	cd ./system; make install
	cd ./integration; make install
	cd ./matricediffusion/;make install

