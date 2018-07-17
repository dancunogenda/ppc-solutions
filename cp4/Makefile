bin=pngcorrelate cp-test cp-benchmark
include ../common/Makefile.openmp

pngcorrelate: pngcorrelate.o cp.o pngio.o error.o vector.o
	$(CXX) $(LDFLAGS) $^ -lpng -o $@

cp-test: cp-test.o cp.o error.o vector.o
	$(CXX) $(LDFLAGS) $^ -o $@

cp-benchmark: cp-benchmark.o cp.o error.o vector.o
	$(CXX) $(LDFLAGS) $^ -o $@

include Makefile.dep
