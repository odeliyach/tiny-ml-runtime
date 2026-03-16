CC      = gcc
CFLAGS  = -Wall -Wextra -O2
LDFLAGS = -lm

.PHONY: all clean test

all: inference

inference: inference.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

test_inference: test_inference.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

test: test_inference
	./test_inference

clean:
	rm -f inference test_inference *.o
