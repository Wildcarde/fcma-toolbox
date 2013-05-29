include ../../Makefile.inc

OBJ=ParallelFor.o
BIN=ParallelFor

#Directory containing OpenCL header files
INCLUDE=-I../../Include -I$(OpenCLInclude)

LIB=-L../.. -lclUtil -lOpenCL
CPPFLAGS=-std=c++0x $(INCLUDE) $(USERFLAGS)

$(BIN): $(OBJ) ../../libclUtil.a
	g++ -o $(BIN) $(OBJ) $(LIB)

.cc.o:
	$(CXX) -c $(CPPFLAGS) -o $@ $<

clean:
	-rm *.o $(BIN)
