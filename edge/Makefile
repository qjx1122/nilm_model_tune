CC ?= cc
CFLAGS ?= -O2 -std=c99 -Wall -Wextra -fPIC -DNILM_EDGE_BUILDING

all: libnilm_edge.so

libnilm_edge.so: nilm_edge.c nilm_edge.h
	$(CC) $(CFLAGS) -shared -o $@ nilm_edge.c -lm

test: all
	python3 ../tests/test_edge_parity.py

clean:
	rm -f libnilm_edge.so

.PHONY: all test clean
