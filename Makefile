# Makefile for loops2 code

#
# C compiler and options for Intel
#
CC=     icc -O3 -qopenmp -std=c99
LIB=    -lm

#
# Object files
#
OBJ=    loops2.o

#
# Compile
#
loops2:   $(OBJ)
	$(CC) -o $@ $(OBJ) $(LIB)

.c.o:
	$(CC) -c $<

#
# Clean out object files and the executable.
#
clean:
	rm *.o loops2
