SRC  = $(wildcard *.c)
OBJ  = $(SRC:.c=.o)
CFLAGS = -pg
EXE  = main

# Cria os Object Files
%.o: %.c
	gcc -c $*.c

#all: $(OBJ)

$(EXE): $(OBJ)
	gcc -o $(EXE) $(OBJ)

.PHONY: clean serial-gprof
clean:
	rm -f $(OBJ) $(EXE)

serial-gprof: $(OBJ)
	gcc -o $(EXE) $(OBJ) $(CFLAGS)
