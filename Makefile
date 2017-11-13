serial:
	cd serial && make && cd ..

openmp:
	cd openmp && make && cd ..

clean:
	cd serial && make clean && cd ..

.PHONY: serial openmp
