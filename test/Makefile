CFLAGS = -O2 -Wall -I../src
CC = gcc

OBJS = ../src/delaunay.o ../src/hash.o \
	../src/istack.o ../src/lpi.o ../src/minell.o ../src/nnai.o \
        ../src/nncommon-vulnerable.o ../src/nncommon.o ../src/nnpi.o \
        ../src/preader.o ../src/triangle.o test.o

TARGETS = all

#
# Rules for making object files from .c files.
#
.c.o:
	$(CC) $(CFLAGS) -o $@ -c $<

all: test

test: $(OBJS) Makefile
	$(CC) $(OBJS) -lm -o $@

clean:
	rm -f $(OBJS) test
