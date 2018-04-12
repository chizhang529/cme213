#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>

#include "tests_q1.h"

uint StringToUint(const std::string& line) {
    std::stringstream buffer;
    uint res;
    buffer << line;
    buffer >> res;
    return res;
}

std::vector<uint> ReadVectorFromFile(const std::string& filename) {
    std::ifstream infile(filename.c_str());

    if(!infile) {
        std::cerr << "Failed to load the file." << std::endl;
    }

    std::vector<uint> res;
    std::string line;

    while(true) {
        getline(infile, line);

        if(infile.fail()) {
            break;
        }

        res.push_back(StringToUint(line));
    }

    return res;
}

