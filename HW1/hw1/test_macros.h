#ifndef TEST_MACROS_H_
#define TEST_MACROS_H_

#define NO_PRETTY_PRINT

#ifdef NO_PRETTY_PRINT
  #define __RESET   ""
  #define __BLACK   ""      /* Black */
  #define __RED     ""      /* Red */
  #define __GREEN   ""      /* Green */
  #define __YELLOW  ""      /* Yellow */
  #define __BLUE    ""      /* Blue */
  #define __MAGENTA ""      /* Magenta */
  #define __CYAN    ""      /* Cyan */
  #define __WHITE   ""      /* White */
  #define __BOLDBLACK   ""      /* Bold Black */
  #define __BOLDRED     ""      /* Bold Red */
  #define __BOLDGREEN   ""      /* Bold Green */
  #define __BOLDYELLOW  ""      /* Bold Yellow */
  #define __BOLDBLUE    ""      /* Bold Blue */
  #define __BOLDMAGENTA ""      /* Bold Magenta */
  #define __BOLDCYAN    ""      /* Bold Cyan */
  #define __BOLDWHITE   ""      /* Bold White */
#else
  // The following are UNIX ONLY terminal color codes.
  // ref: http://stackoverflow.com/questions/9158150/colored-output-in-c
  #define __RESET   "\033[0m"
  #define __BLACK   "\033[30m"      /* Black */
  #define __RED     "\033[31m"      /* Red */
  #define __GREEN   "\033[32m"      /* Green */
  #define __YELLOW  "\033[33m"      /* Yellow */
  #define __BLUE    "\033[34m"      /* Blue */
  #define __MAGENTA "\033[35m"      /* Magenta */
  #define __CYAN    "\033[36m"      /* Cyan */
  #define __WHITE   "\033[37m"      /* White */
  #define __BOLDBLACK   "\033[1m\033[30m"      /* Bold Black */
  #define __BOLDRED     "\033[1m\033[31m"      /* Bold Red */
  #define __BOLDGREEN   "\033[1m\033[32m"      /* Bold Green */
  #define __BOLDYELLOW  "\033[1m\033[33m"      /* Bold Yellow */
  #define __BOLDBLUE    "\033[1m\033[34m"      /* Bold Blue */
  #define __BOLDMAGENTA "\033[1m\033[35m"      /* Bold Magenta */
  #define __BOLDCYAN    "\033[1m\033[36m"      /* Bold Cyan */
  #define __BOLDWHITE   "\033[1m\033[37m"      /* Bold White */
#endif

#define PRINT_SUCCESS(success) { \
  if (success) { \
    std::cout << __FILE__ << ":" << __LINE__ << ":" << __BLUE << __func__ \
              << __GREEN << "\tTEST PASSED." << __RESET << std::endl; \
  } else { \
    std::cout << __FILE__ << ":" << __LINE__ << ":" << __BLUE << __func__ \
              << __RED << "\tTEST FAILED." << __RESET << std::endl; \
  } \
}

#define EXPECT_EQ_EPS(a, b, eps, success) { \
  if (a - b > eps or b - a > eps) { \
    std::cerr << __FILE__ << ":" << __LINE__ << "\t" << __RED \
              << "\n\tERROR:" << __RESET  << "Value Mismatch (" << a \
              << " != " << b << ")" << std::endl; \
    *success = false; \
  } \
}

#define EXPECT_EQ(a, b, success) { \
  EXPECT_EQ_EPS(a, b, 0, success) \
}

#define EXPECT_VECTOR_EQ_EPS(ref_vec, test_vec, eps, success) { \
  if (ref_vec.size() != test_vec.size()) { \
    std::cerr << __FILE__ << ":" << __LINE__ << "\t" << __RED \
              << "\n\tERROR:" << __RESET << "Dimension Mismatch (" \
              << ref_vec.size() << " != " << test_vec.size() << ")" \
              << std::endl; \
    *success = false; \
  } else { \
    for (unsigned int i = 0; i < ref_vec.size(); ++i) { \
      if (ref_vec[i] - test_vec[i] > eps or test_vec[i] - ref_vec[i] > eps) { \
        std::cerr << __FILE__ << ":" << __LINE__ << "\t" << __RED \
                  << "\n\tERROR: " << __RESET << "Value Mismatch (" \
                  << ref_vec[i] << " != " << test_vec[i] << ", index = " << i \
                  << ")" << std::endl; \
        *success = false; \
        break; \
      } \
    } \
  } \
}

#define EXPECT_VECTOR_EQ(ref_vec, test_vec, success) { \
  EXPECT_VECTOR_EQ_EPS(ref_vec, test_vec, 0, success); \
}

#endif /* TEST_MACROS_H_ */
