CC ?= cc
CFLAGS ?= -O2 -std=c99 -Wall -Wextra -fPIC -DNILM_EDGE_BUILDING

all: libnilm_edge.so

libnilm_edge.so: nilm_edge.c nilm_edge.h
	$(CC) $(CFLAGS) -shared -o $@ nilm_edge.c -lm

test: all
	python3 ../tests/test_edge_parity.py

clean:
	rm -f libnilm_edge.so librecorder.so recorder.dll

.PHONY: all test clean recorder

recorder: librecorder.so recorder.dll

librecorder.so: recorder.c recorder.h
	$(CC) $(CFLAGS) -DNILM_REC_BUILDING -shared -o $@ recorder.c

recorder.dll: recorder.c recorder.h
	python -m ziglang cc -target x86_64-windows-gnu -O2 -std=c99 -fPIC -shared -DNILM_REC_BUILDING recorder.c -o $@
