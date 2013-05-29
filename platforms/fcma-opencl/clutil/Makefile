include Makefile.inc

OBJ=clUtil.o clUtilKernel.o clUtilInit.o clUtilDevice.o
OBJ+=clUtilPlatform.o clUtilUtility.o clUtilDeviceHelpers.o clUtilMemory.o
OBJ+=clUtilParallelFor.o clUtilDeviceGroup.o
OBJ+=Scheduler/StaticScheduler.o Scheduler/EGSScheduler.o 
#OBJ+=Scheduler/PINAScheduler.o
EXAMPLES=$(wildcard examples/*) #Every subdir in examples

#Directory containing OpenCL header files

LIB=
#CPPFLAGS=-std=c++0x -I$(OpenCLInclude) $(USERFLAGS) -IInclude
CPPFLAGS=-std=c++11 -stdlib=libc++ -I$(OpenCLInclude) $(USERFLAGS) -IInclude

libclUtil.a: $(OBJ)
	$(AR) rcs libclUtil.a $(OBJ) $(LIB)

all: libclUtil.a
	make examples

examples: libclUtil.a
	for dir in $(EXAMPLES); do make -C $$dir; done

.cc.o:
	$(CXX) -c $(CPPFLAGS) -o $@ $<

install: libclUtil.a
	mkdir -p /usr/include/clutil
	cp -Rf Include/* /usr/include/clutil
	cp -f libclUtil.a /usr/lib

uninstall:
	rm -Rf /usr/include/clutil
	rm /usr/lib/libclUtil.a

clean:
	-rm *.o libclUtil.a
	for dir in $(EXAMPLES); do make -C $$dir clean; done
