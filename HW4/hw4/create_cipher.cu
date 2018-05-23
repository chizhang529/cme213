#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <vector>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <thrust/inner_product.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include "test_macros.h"

// You will need to call these functors from
// thrust functions in the code do not create new ones

// returns true if the char is not a lowercase letter
struct isnot_lowercase_alpha : thrust::unary_function<unsigned char, bool> {
    __host__ __device__
    bool operator()(const unsigned char c) const {
        return (c < 'a') || (c > 'z');
    }
};

// convert an uppercase letter into a lowercase one
// do not use the builtin C function or anything from boost, etc.
struct upper_to_lower : thrust::unary_function<unsigned char, unsigned char> {
    __host__ __device__
    unsigned char operator()(const unsigned char c) const {
        if ((c >= 'A') && (c <= 'Z')) {
            return (c | 0x20);
        } else {
            return c;
        }
    }
};

// apply a shift with appropriate wrapping (assume the input char is in the range ['a', 'z'])
struct apply_shift : thrust::binary_function<unsigned char, int,
        unsigned char> {
    thrust::device_ptr<unsigned int> shift_arr;
    const unsigned int period;
    // constructor
    apply_shift(thrust::device_ptr<unsigned int> arr, int _period):
                shift_arr(arr), period(_period) {}

    __host__ __device__
    unsigned char operator()(const unsigned char c, const int pos) const {
        const unsigned int shift = shift_arr[pos % period];
        const unsigned char num_letters = 'z' - 'a' + 1;
        const unsigned char offset = (c - 'a' + shift) % num_letters;
        return (c + offset);
    }
};

// Returns a vector with the top 5 letter frequencies in text.
std::vector<double> getLetterFrequencyCpu(
    const std::vector<unsigned char>& text) {
    std::vector<unsigned int> freq(256);

    for(unsigned int i = 0; i < text.size(); ++i) {
        freq[tolower(text[i])]++;
    }

    unsigned int sum_chars = 0;

    for(unsigned char c = 'a'; c <= 'z'; ++c) {
        sum_chars += freq[c];
    }

    std::vector<double> freq_alpha_lower;

    for(unsigned char c = 'a'; c <= 'z'; ++c) {
        if(freq[c] > 0) {
            freq_alpha_lower.push_back(freq[c] / static_cast<double>(sum_chars));
        }
    }

    std::sort(freq_alpha_lower.begin(), freq_alpha_lower.end(),
              std::greater<double>());
    freq_alpha_lower.resize(min(static_cast<int>(freq_alpha_lower.size()), 5));

    return freq_alpha_lower;
}

// Print the top 5 letter frequencies and them.
std::vector<double> getLetterFrequencyGpu(const thrust::device_vector<unsigned char>& text) {
    /* NOTE: assume input only contains ascii characters but the histogram may be very sparse
             for cases like 'xxxxxxxxxy', 'aabbcc', etc. */
    // copy and sort data
    thrust::device_vector<unsigned char> data(text);
    thrust::sort(data.begin(), data.end());
    // count the number of bins needed
    const unsigned int num_bins = thrust::inner_product(data.begin(), data.end() - 1,
                                                        data.begin() + 1,
                                                        1,
                                                        thrust::plus<int>(),
                                                        thrust::not_equal_to<unsigned char>());

    thrust::device_vector<unsigned char> letters(num_bins);
    thrust::device_vector<double> freq(num_bins);
    // calculate letter frequency
    const double denom = 1.0 / (double)text.size();
    thrust::reduce_by_key(data.begin(), data.end(),
                          thrust::constant_iterator<double>(denom),
                          letters.begin(), freq.begin());
    // sort based on frequency (descending order)
    thrust::sort_by_key(freq.begin(), freq.end(),
                        letters.begin(), thrust::greater<double>());

    // copy the result to a standard vector
    std::vector<double> freq_alpha_lower(num_bins);
    thrust::copy(freq.begin(), freq.end(), freq_alpha_lower.begin());
    // resize the vector to take into account cases where less than 5 letters appearing
    freq_alpha_lower.resize(min(static_cast<int>(freq_alpha_lower.size()), 5));
    // print result
    size_t sz = freq_alpha_lower.size();
    std::cout << "Top " << sz << " Letter Frequencies" << std::endl
              << "-------------" << std::endl;
    for(int i = 0; i < sz; ++i)
        std::cout << letters[i] << ": " << freq_alpha_lower[i] << std::endl;

    return freq_alpha_lower;
}

int main(int argc, char** argv) {
    if(argc != 3) {
        std::cerr << "Didn't supply plain text and period!" << std::endl;
        return 1;
    }

    std::ifstream ifs(argv[1], std::ios::binary);

    if(!ifs.good()) {
        std::cerr << "Couldn't open text file!" << std::endl;
        return 1;
    }

    unsigned int period = atoi(argv[2]);

    if(period < 4) {
        std::cerr << "Period must be at least 4!" << std::endl;
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

    // BEGIN YOUR CODE
    thrust::device_vector<unsigned char> text_clean(length);
    // sanitize input to contain only a-z lowercase (use the
    // isnot_lowercase_alpha functor), calculate the number of characters
    // in the cleaned text and put the result in text_clean, make sure to
    // resize text_clean to the correct size
    thrust::device_vector<unsigned char> d_text(text);
    thrust::device_vector<unsigned char>::iterator end_iter =
    thrust::remove_copy_if(
        thrust::make_transform_iterator(d_text.begin(), upper_to_lower()),
        thrust::make_transform_iterator(d_text.end(), upper_to_lower()),
        text_clean.begin(),
        isnot_lowercase_alpha()
    );

    int numElements = thrust::distance(text_clean.begin(), end_iter);
    text_clean.resize(numElements);
    // END YOUR CODE

    std::cout << "\nBefore ciphering!" << std::endl << std::endl;
    std::vector<double> letterFreqGpu = getLetterFrequencyGpu(text_clean);
    std::vector<double> letterFreqCpu = getLetterFrequencyCpu(text);
    bool success = true;
    EXPECT_VECTOR_EQ_EPS(letterFreqCpu, letterFreqGpu, 1e-14, &success);
    PRINT_SUCCESS(success);

    // BEGIN YOUR CODE
    thrust::device_vector<unsigned int> shifts(period);
    // fill in shifts using thrust random number generation (make sure
    // not to allow 0-shifts, this would make for rather poor encryption)
    thrust::default_random_engine srand(123);
    thrust::uniform_int_distribution<int> radnom_shift(1, 'z' - 'a');
    for (size_t i = 0; i < period; ++i)
        shifts[i] = radnom_shift(srand);
    // END YOUR CODE

    std::cout << "\nEncryption key: ";
    for(int i = 0; i < period; ++i) {
        std::cout << static_cast<char>('a' + shifts[i]);
    }
    std::cout << std::endl;

    // BEGIN YOUR CODE
    thrust::device_vector<unsigned char> device_cipher_text(numElements);
    // apply the shifts to text_clean and place the result in device_cipher_text
    thrust::transform(text_clean.begin(), text_clean.end(),
                      thrust::make_counting_iterator(0), device_cipher_text.begin(),
                      apply_shift(shifts.data(), period));
    // END YOUR CODE

    thrust::host_vector<unsigned char> host_cipher_text = device_cipher_text;

    std::cout << "After ciphering!" << std::endl << std::endl;
    getLetterFrequencyGpu(device_cipher_text);

    std::ofstream ofs("cipher_text.txt", std::ios::binary);

    ofs.write((char*)&host_cipher_text[0], numElements);

    ofs.close();

    return 0;
}
