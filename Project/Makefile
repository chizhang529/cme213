HOMEDIR = $(shell pwd)
OBJDIR=$(HOMEDIR)/obj
LIBDIR=$(HOMEDIR)/lib
INCDIR=$(HOMEDIR)/inc

# Compilers
CC=mpic++
CUD=nvcc

# Flags
CFLAGS=-O2 -std=c++11 -I$(LIBDIR)/usr/include
LFLAGS= -O2 -larmadillo -lcublas -lcudart -L=$(LIBDIR)/usr/lib64 -L=/usr/local/cuda-8.0/lib64 -Wl,-rpath=$(LIBDIR)/usr/lib64
CUDFLAGS=-O2 -c -arch=sm_20 -Xcompiler -Wall,-Winline,-Wextra,-Wno-strict-aliasing 
INCFLAGS=-I/usr/local/cuda-8.0/include -I/usr/local/cuda-8.0/samples/common/inc -I$(INCDIR)
#-fmad=false

main:  mnist.o tests.o common.o gpu_func.o neural_network.o main.o 
	cd $(OBJDIR); $(CC) $(LFLAGS) $(INCFLAGS) main.o neural_network.o mnist.o common.o gpu_func.o tests.o -o ../main

main.o: main.cpp utils/test_utils.h $(INCDIR)/neural_network.h
	$(CC) $(CFLAGS) $(INCFLAGS)  -c main.cpp -o $(OBJDIR)/main.o

neural_network.o: neural_network.cpp $(INCDIR)/neural_network.h utils/test_utils.h
	$(CC) $(CFLAGS) $(INCFLAGS)  -c neural_network.cpp -o $(OBJDIR)/neural_network.o

mnist.o: utils/mnist.cpp
	$(CC) $(CFLAGS) $(INCFLAGS)  -c utils/mnist.cpp -o $(OBJDIR)/mnist.o

tests.o: utils/tests.cpp utils/tests.h
	$(CC) $(CFLAGS) $(INCFLAGS)  -c utils/tests.cpp -o $(OBJDIR)/tests.o

common.o: utils/common.cpp
	$(CC) $(CFLAGS) $(INCFLAGS)  -c utils/common.cpp -o $(OBJDIR)/common.o

gpu_func.o: gpu_func.cu
	$(CUD) $(CUDFLAGS) $(INCFLAGS)  -c gpu_func.cu -o $(OBJDIR)/gpu_func.o

clean:
	rm -rf $(OBJDIR)/*.o main

clear:
	rm -rf  cme213.* 
