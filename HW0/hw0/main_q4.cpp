#include <iostream>
#include <cassert>
#include <set>
#include <random>

size_t naiveCount(std::set<double> &data, const double lb, const double ub)
{
    if (data.empty()) return 0;

    size_t count = 0;
    for (auto num : data) {
        if (num >= lb && num <= ub)
            count++;
    }

    return count;
}

size_t dataPointNumber(std::set<double> &data, const double lb, const double ub)
{
    if (data.empty()) return 0;

    std::set<double>::iterator it_lb = data.lower_bound(lb);
    std::set<double>::iterator it_ub = data.upper_bound(ub);

    return std::distance(it_lb, it_ub);
}

int main(int argc, char **argv)
{
    // generate random data
    std::set<double> data;
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, 1.0);
    for (unsigned int i = 0; i < 1000; ++i)
        data.insert(distribution(generator)); // sorted numbers

    const double lb = 2, ub = 10;
    size_t num = dataPointNumber(data, lb, ub);
    // sanity check
    assert(num == naiveCount(data, lb, ub));

    std::cout << "The number of data points in range [" << lb << ", " << ub << "] "
              << "is " << num << std::endl;
    return 0;
}
