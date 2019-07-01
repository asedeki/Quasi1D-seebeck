all : libsigma.so


libsigma.so  : sigma.o
	#ar cr $@ $^
	clang -lgsl -shared -w -o $@ $^

%.o: %.c %.h
	gcc -c -fPIC $< -o $@

clean :
	-rm -vf *.o *.so
	netois
