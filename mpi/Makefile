SRC  = $(wildcard *.c)
OBJ  = $(SRC:.c=.o)
EXE  = main
CC   = mpicc

# Cria os Object Files
%.o: %.c
	$(CC) -c -O3 -Wall -std=c99 $*.c -fopenmp

#all: $(OBJ)

$(EXE): $(OBJ)
	$(CC) -o $(EXE) $(OBJ) -O3 -lm -fopenmp

.PHONY: clean
clean:
	rm -f $(OBJ) $(OBJ_GPROF) $(EXE)
