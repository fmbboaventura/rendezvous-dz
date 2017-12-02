SRC    = $(wildcard *.c)
OBJ    = $(SRC:.c=.o)
EXE    = main
CCFLAGS = -Wall -std=c99
ifeq ($(OS),Windows_NT)
	LIBS = C:\Windows\System32\OpenCL.dll
else
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Darwin)
		LIBS = -framework OpenCL
	endif
endif

main: main.c wtime.c device_info.c
	gcc $^ $(CCFLAGS) $(LIBS) -o $@

.PHONY: clean
clean:
	rm -f $(OBJ) $(OBJ_GPROF) $(EXE)
