include ../../Makefile.inc

OBJ=Sort.o
BIN=Sort

#Directory containing OpenCL header files
INCLUDE=-I../../Include -I$(OpenCLInclude)

LIB=-lOpenCL -L../.. -lclUtil
CPPFLAGS=-std=c++0x $(INCLUDE) $(USERFLAGS)

$(BIN): $(OBJ)
	g++ -o $(BIN) $(OBJ) $(LIB)

.cc.o:
	$(CXX) -c $(CPPFLAGS) -o $@ $<

clean:
	-rm *.o $(BIN)
