bin=pngmf mf-test mf-benchmark
include ../common/Makefile.common

pngmf: pngmf.o mf.o pngio.o error.o
	$(CXX) $(LDFLAGS) $^ -lpng -o $@

mf-test: mf-test.o mf.o error.o
	$(CXX) $(LDFLAGS) $^ -o $@

mf-benchmark: mf-benchmark.o mf.o error.o
	$(CXX) $(LDFLAGS) $^ -o $@

include Makefile.dep
