all: cubatureintegration.so

cubatureintegration.so: cubatureintegration.pxd cubatureintegration.pyx
	python3 setup.py build_ext -i

clean:
	rm -f *.html *.c 
	rm -rf build/

cleanall:
	rm -f *.html *.c
	rm -rf build/
	rm -f *.so
install:
	mv *.so ../../lib
	cp *.pxd ../../lib
