CXX=g++
CUD=nvcc
LDFLAGS=-lcudart -L=/usr/local/cuda-8.0/lib64
CXXFLAGS=-O3 -Wall -Winline -Wextra -Wno-strict-aliasing
CUDFLAGS=-O3 -arch=sm_20 -Xcompiler -Wall,-Winline,-Wextra,-Wno-strict-aliasing


main: main.o simParams.o Grid.o
	$(CXX) $^ $(LDFLAGS) $(CXXFLAGS) -o $@

main.o: main.cu mp1-util.h simParams.h Grid.h gpuStencil.cu BC.h
	$(CUD) -c $< $(CUDFLAGS)

simParams.o: simParams.cpp simParams.h
	$(CXX) -c $< $(CXXFLAGS)

Grid.o: Grid.cu Grid.h
	$(CUD) -c $< $(CUDFLAGS)

clean:
	rm -f *.o *~ *~ *Errors.txt main 
	rm -rf *.dSYM
