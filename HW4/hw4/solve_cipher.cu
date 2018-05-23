#include <vector>
#include <fstream>
#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/extrema.h>

#include "strided_range_iterator.h"

// You will need to call these functors from thrust functions in the code
// do not create new ones

// this can be the same as in create_cipher.cu
struct apply_shift : thrust::binary_function<unsigned char, int,
        unsigned char> {
    thrust::device_ptr<int> shift_arr;
    const unsigned int period;
    // constructor
    apply_shift(thrust::device_ptr<int> arr, int _period):
                shift_arr(arr), period(_period) {}

    __host__ __device__
    unsigned char operator()(const unsigned char c, const int pos) const {
        const unsigned char num_letters = 'z' - 'a' + 1;
        const int shift = shift_arr[pos % period] + num_letters; // WARNING: shift could be negative now
        const unsigned char offset = (c - 'a' + shift) % num_letters;
        return ('a' + offset);
    }
};

int main(int argc, char** argv) {
    if(argc != 2) {
        std::cerr << "No cipher text given!" << std::endl;
        return 1;
    }

    // First load the text
    std::ifstream ifs(argv[1], std::ios::binary);

    if(!ifs.good()) {
        std::cerr << "Couldn't open book file!" << std::endl;
        return 1;
    }

    // load the file into text
    std::vector<unsigned char> text;

    ifs.seekg(0, std::ios::end); // seek to end of file
    int length = ifs.tellg();    // get distance from beginning
    ifs.seekg(0, std::ios::beg); // move back to beginning

    text.resize(length);
    ifs.read((char*)&text[0], length);

    ifs.close();

    // we assume the cipher text has been sanitized
    thrust::device_vector<unsigned char> text_clean = text;

    // now we crack the Vigenere cipher
    // first we need to determine the key length
    // use the kappa index of coincidence
    int keyLength = 0;
    {
        bool found = false;
        int shift_idx = 4; // Start at index 4.

        while(!found) {
            // compute the number of characters that match
            // when shifting text_clean by shift_idx.
            int numMatches = thrust::inner_product(text_clean.begin(),
                                                   text_clean.end() - shift_idx,
                                                   text_clean.begin() + shift_idx,
                                                   0,
                                                   thrust::plus<int>(),
                                                   thrust::equal_to<unsigned char>());

            double ioc = numMatches /
                         static_cast<double>((text_clean.size() - shift_idx) / 26.);

            std::cout << "Period " << shift_idx << " ioc: " << ioc << std::endl;

            if(ioc > 1.6) {
                if(keyLength == 0) {
                    keyLength = shift_idx;
                    shift_idx = 2 * shift_idx - 1; // check double the period to make sure
                } else if(2 * keyLength == shift_idx) {
                    found = true;
                } else {
                    std::cout << "Unusual pattern in text!" << std::endl;
                    exit(1);
                }
            }

            ++shift_idx;
        }
    }

    std::cout << "keyLength: " << keyLength << std::endl;

    // once we know the key length, then we can do frequency analysis on each
    // pos mod length allowing us to easily break each cipher independently
    // you will find the strided_range useful
    // it is located in strided_range_iterator.h and an example
    // of how to use it is located in the that file
    thrust::device_vector<unsigned char> text_copy = text_clean;
    thrust::device_vector<int> dShifts(keyLength);
    typedef thrust::device_vector<unsigned char>::iterator Iterator;

    // now that you have determined the length of the key, you need to
    // compute the actual key. To do so, perform keyLength individual frequency
    // analyses on text_copy to find the shift which aligns the most common
    // character in text_copy with the character 'e'. Fill up the
    // dShifts vector with the correct shifts.
    thrust::device_vector<int> record(26);
    for (size_t i = 0; i < keyLength; ++i) {
        strided_range<Iterator> sequence(text_copy.begin() + i, text_copy.end(), keyLength);
        // copy and sort data
        thrust::sort(sequence.begin(), sequence.end());
        // count the number of bins needed
        const unsigned int num_bins = thrust::inner_product(sequence.begin(),
                                                            sequence.end() - 1,
                                                            sequence.begin() + 1,
                                                            1,
                                                            thrust::plus<int>(),
                                                            thrust::not_equal_to<unsigned char>());

        thrust::device_vector<unsigned char> letters(num_bins);
        thrust::device_vector<unsigned int> freq(num_bins);
        // calculate letter frequency
        thrust::reduce_by_key(sequence.begin(), sequence.end(),
                              thrust::constant_iterator<int>(1),
                              letters.begin(), freq.begin());
        // sort based on frequency (descending order)
        thrust::sort_by_key(freq.begin(), freq.end(),
                            letters.begin(), thrust::greater<unsigned int>());

        const unsigned char common_char = letters[0];
        const int shift = 'e' - common_char;
        dShifts[i] = shift;
    }

    std::cout << "\nEncryption key: ";

    for(unsigned int i = 0; i < keyLength; ++i)
        std::cout << static_cast<char>('a' - (dShifts[i] <= 0 ? dShifts[i] :
                                              dShifts[i] - 26));

    std::cout << std::endl;

    // take the shifts and transform cipher text back to plain text
    // transform the cipher text back to the plain text by using the
    // apply_shift functor
    thrust::transform(text_clean.begin(), text_clean.end(),
                      thrust::make_counting_iterator(0), text_clean.begin(),
                      apply_shift(dShifts.data(), keyLength));

    thrust::host_vector<unsigned char> h_plain_text = text_clean;

    std::ofstream ofs("plain_text.txt", std::ios::binary);
    ofs.write((char*)&h_plain_text[0], h_plain_text.size());
    ofs.close();

    return 0;
}
