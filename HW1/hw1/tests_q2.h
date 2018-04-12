#ifndef TESTS_Q2_H
#define TESTS_Q2_H

#include <iostream>
#include <string>
#include <vector>

#include "test_macros.h"

typedef unsigned int uint;

void TestCorrectness(const std::vector<uint>& reference, const std::vector<uint>& toTest);

void WriteVectorToFile(const std::string& filename, std::vector<uint>& v);

std::vector<uint> ReadVectorFromFile(const std::string& filename);

uint StringToUint(const std::string& line);

void Test1();

void Test2();

void Test3();

void Test4();

void Test5();

#endif /* TESTS_H */
