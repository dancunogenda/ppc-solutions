bin=pngcorrelate cp-test cp-benchmark
include ../common/Makefile.cuda

cp.o: cp.cu
	$(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

pngcorrelate: pngcorrelate.o cp.o pngio.o error.o
	$(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ -lpng

cp-test: cp-test.o cp.o error.o
	$(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+

cp-benchmark: cp-benchmark.o cp.o error.o
	$(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+

include Makefile.dep
