bin=so-test
include ../common/Makefile.openmp

so-test: so-test.o so.o error.o
	$(CXX) $(LDFLAGS) $^ -o $@

include Makefile.dep
