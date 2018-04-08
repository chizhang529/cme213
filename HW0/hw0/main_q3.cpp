#include <iostream>
#include <string>
#include <vector>

/* TODO: Make Matrix a pure abstract class with the 
 * public method:
 *     std::string repr()
 */
class Matrix {
public:
	virtual std::string repr() = 0;	
};

/* TODO: Modify the following function so that it 
 * inherits from the Matrix class */
class SparseMatrix : public Matrix {
public:
	std::string repr() {
		return "sparse";
	}
};

/* TODO: Modify the following function so that it 
 * inherits from the Matrix class */
class ToeplitzMatrix : public Matrix {
public:
	std::string repr() {
		return "toeplitz";
	}
};

/* TODO: This function should accept a vector of Matrices and call the repr 
 * function on each matrix, printing the result to standard out. 
 */
void PrintRepr(std::vector<Matrix *> &matVec)
{
	for (auto mat : matVec)
		std::cout << mat->repr() << std::endl;
}

/* TODO: Your main method should fill a vector with an instance of SparseMatrix
 * and an instance of ToeplitzMatrix and pass the resulting vector
 * to the PrintRepr function.
 */ 
int main()
{ 
	std::vector<Matrix *> matVec;
	// add matrix instances
	matVec.push_back(new SparseMatrix());
	matVec.push_back(new ToeplitzMatrix());

	PrintRepr(matVec);
	
	return 0;
}

