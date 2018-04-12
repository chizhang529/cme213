CXX=g++
CXXFLAGS=-std=c++11 -g -Wall -O3 -fopenmp
SRC1=main_q1.cpp tests_q1.cpp
SRC2=main_q2.cpp tests_q2.cpp
INC1=tests_q1.h
INC2=test_macros.h tests_q2.h

default: main_q1 main_q2 main_q2_part6
main_q1: $(INC1) $(SRC1)
	$(CXX) $(CXXFLAGS) $(SRC1) $< -o $@
main_q2: $(INC2) $(SRC2)
	$(CXX) $(CXXFLAGS) $(SRC2) $< -o $@
main_q2_part6: $(INC2) $(SRC2)
	$(CXX) $(CXXFLAGS) $(SRC2) $< -D QUESTION6 -o $@

clean:
	rm -f *.o *~ *~ main_q1 main_q2 main_q2_part6
	rm -rf *.dSYM
