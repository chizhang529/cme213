CUD=nvcc
CUDFLAGS=-O3 -arch=sm_20 

all: create_cipher solve_cipher

create_cipher: create_cipher.cu
	$(CUD) -o $@ $< $(CUDFLAGS)

solve_cipher: solve_cipher.cu
	$(CUD) -o $@ $< $(CUDFLAGS)

clean:
	rm -f *.o *~ *~ create_cipher solve_cipher
	rm -rf *.dSYM
