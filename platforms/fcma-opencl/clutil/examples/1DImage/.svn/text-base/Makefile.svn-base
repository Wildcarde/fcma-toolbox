include ../../Makefile.inc

OBJ=1DImage.o

#Directory containing OpenCL header files
INCLUDE=-I../../Include -I$(OpenCLInclude)

LIB=-lOpenCL -L../.. -lclUtil
CPPFLAGS=-std=c++0x $(INCLUDE) $(USERFLAGS)

1DImage: $(OBJ) ../../libclUtil.a
	g++ -o 1DImage $(OBJ) $(LIB)

.cc.o:
	$(CXX) -c $(CPPFLAGS) -o $@ $<

clean:
	-rm *.o 1DImage
