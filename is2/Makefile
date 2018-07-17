bin=pngsegment is-test is-benchmark
include ../common/Makefile.openmp

pngsegment: pngsegment.o is.o pngio.o error.o vector.o
	$(CXX) $(LDFLAGS) $^ -lpng -o $@

is-test: is-test.o is.o error.o vector.o
	$(CXX) $(LDFLAGS) $^ -o $@

is-benchmark: is-benchmark.o is.o error.o vector.o
	$(CXX) $(LDFLAGS) $^ -o $@

include Makefile.dep
