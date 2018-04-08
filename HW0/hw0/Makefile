CXX=g++
CXXFLAGS=-std=c++11 -g -Wall
SRC1=main_q1.cpp
SRC2=main_q2.cpp
SRC3=main_q3.cpp
SRC4=main_q4.cpp
HDR= matrix.hpp matrix_lt.hpp

default: main_q1 main_q2 main_q3 main_q4
main_q1: $(SRC1)
	$(CXX) $(CXXFLAGS) $< -o $@
main_q2: $(SRC2) $(HDR)
	$(CXX) $(CXXFLAGS) $< -o $@
main_q3: $(SRC3)
	$(CXX) $(CXXFLAGS) $< -o $@
main_q4: $(SRC4)
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f *.o *~ *~ main_q1 main_q2 main_q3 main_q4
	rm -rf *.dSYM

