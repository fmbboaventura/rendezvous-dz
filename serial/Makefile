SRC  = $(wildcard *.c)
OBJ  = $(SRC:.c=.o)
OBJ_GPROF = $(SRC:.c=.gprof.o)
EXE  = main

# Cria os Object Files
%.o: %.c
	gcc -c $*.c

%.gprof.o: %.c
	gcc -c $*.c -pg -o $*.gprof.o

#all: $(OBJ)

$(EXE): $(OBJ)
	gcc -o $(EXE) $(OBJ) -lm

.PHONY: clean serial-gprof
clean:
	rm -f $(OBJ) $(OBJ_GPROF) $(EXE)

serial-gprof: $(OBJ_GPROF)
	gcc -o $(EXE) $(OBJ_GPROF) -lm -pg
