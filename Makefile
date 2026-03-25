CC      = gcc
CFLAGS  = -Wall -Wextra -O2
LDFLAGS = -lm

ifeq ($(COVERAGE),1)
  CFLAGS  += --coverage -g -O0
  LDFLAGS += --coverage
endif

.PHONY: all clean test c_tests test_unity

all: inference

inference: src/c/inference.c src/c/inference.h
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

test_inference: src/c/test_inference.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

tests/c/test_inference_unity: tests/c/test_inference_unity.c src/c/inference.c src/c/inference.h tests/c/unity/unity.c tests/c/unity/unity.h
	$(CC) $(CFLAGS) -I./src/c -I./tests/c/unity -DTINY_ML_INFERENCE_NO_MAIN -o $@ tests/c/test_inference_unity.c src/c/inference.c tests/c/unity/unity.c $(LDFLAGS)

c_tests: test_inference tests/c/test_inference_unity
	./test_inference
	./tests/c/test_inference_unity

test: c_tests

clean:
	rm -f inference test_inference tests/c/test_inference_unity *.o
	rm -f *.gcno *.gcda coverage.info
