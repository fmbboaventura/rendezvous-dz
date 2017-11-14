serial:
	cd serial && make && cd ..

serial-gprof:
	cd serial && make serial-gprof && cd ..

openmp:
	cd openmp && make && cd ..

clean:
	cd serial && make clean && cd ..

.PHONY: serial serial-gprof openmp
