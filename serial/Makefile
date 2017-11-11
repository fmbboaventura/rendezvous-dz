SRC  = $(wildcard *.c)
OBJ  = $(SRC:.c=.o)
EXE  = main

# Cria os Object Files
%.o: %.c
	gcc -c $*.c

#all: $(OBJ)

$(EXE): $(OBJ)
	gcc -o $(EXE) $(OBJ)

.PHONY: clean
clean:
	rm -f $(OBJ)
