include ../../Makefile.inc

DGEMMOBJ=xgemm.o dgemm_main.o Matrix.o ../../libclUtil.a
SGEMMOBJ=xgemm.o sgemm_main.o Matrix.o ../../libclUtil.a
CXXFLAGS+=$(USERFLAGS) -std=c++0x -I$(OpenCLInclude) -I../../Include
LDFLAGS=-lOpenCL 

all:
	make dgemm
	make sgemm

dgemm: $(DGEMMOBJ)
	$(CXX) $(DGEMMOBJ) -o dgemm $(LDFLAGS)

sgemm: $(SGEMMOBJ)
	$(CXX) $(SGEMMOBJ) -o sgemm $(LDFLAGS)

.cc.o:
	$(CXX) -c $(CXXFLAGS) -o $@ $<

xgemm.o: dgemm.cl sgemm.cl
	$(CXX) -c $(CXXFLAGS) -o xgemm.o xgemm.cc

dgemm.cl: gen_dgemm.py
	python gen_dgemm.py > dgemm.cl

sgemm.cl: gen_sgemm.py
	python gen_sgemm.py > sgemm.cl

clean:
	-rm *.o sgemm dgemm dgemm.cl sgemm.cl clUtilProfile.svg
