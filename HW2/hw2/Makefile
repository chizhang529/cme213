CXX=nvcc
CXXFLAGS=-g --compiler-options -Wall -O3 -arch=sm_20
SRC1=main_q1.cu
SRC2=main_q2.cu
INC=common

default: main_q1 main_q2
main_q1: $(SRC1)
	$(CXX) $(CXXFLAGS) $(SRC1) -I $(INC) -o $@
main_q2: $(SRC2)
	$(CXX) $(CXXFLAGS) $(SRC2) -I $(INC) -o $@

clean:
	rm -f *.o *~ *~ main_q1 main_q2
	rm -rf *.dSYM
