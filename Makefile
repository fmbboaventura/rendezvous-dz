serial:
	cd serial && make && cd ..

serial-gprof:
	cd serial && make serial-gprof && cd ..

openmp:
	cd openmp && make && cd ..

opencl:
	cd opencl && make && cd ..

clean:
	cd serial && make clean && cd ..
	cd openmp && make clean && cd ..
	cd opencl && make clean && cd ..

.PHONY: serial serial-gprof openmp opencl
