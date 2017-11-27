SRC  = $(wildcard *.c)
OBJ  = $(SRC:.c=.o)
EXE  = main

# Cria os Object Files
%.o: %.c
	gcc -c -Wall $*.c -fopenmp

#all: $(OBJ)

$(EXE): $(OBJ)
	gcc -o $(EXE) $(OBJ) -lm -fopenmp

.PHONY: clean
clean:
	rm -f $(OBJ) $(OBJ_GPROF) $(EXE)
