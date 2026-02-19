#include <iostream>
#include <cassert>
#include <cmath>
#include <algorithm>
#include "ds/statistics.hpp"
#include "ds/linear_algebra.hpp"

using namespace ds;

// Helper function to check floating point equality
bool approx_equal(double a, double b, double epsilon = 1e-6) {
    return std::abs(a - b) < epsilon;
}

// Helper to check vector equality
bool vectors_equal(const Vector& v1, const Vector& v2, double epsilon = 1e-6) {
    if (v1.size() != v2.size()) return false;
    for (size_t i = 0; i < v1.size(); ++i) {
        if (!approx_equal(v1[i], v2[i], epsilon)) return false;
    }
    return true;
}

// Helper to print a vector
void print_vector(const Vector& v, const std::string& name = "") {
    if (!name.empty()) std::cout << name << ": ";
    std::cout << "[ ";
    for (double x : v) std::cout << x << " ";
    std::cout << "]\n";
}

// ============== Statistics Tests ==============

void test_mean() {
    std::cout << "\n--- Testing mean ---\n";
    Vector v{1.0, 2.0, 3.0, 4.0, 5.0};
    double result = mean(v);
    double expected = 3.0;
    assert(approx_equal(result, expected) && "mean failed");
    std::cout << "✓ mean({1,2,3,4,5}) = " << result << "\n";
}

void test_median_odd() {
    std::cout << "\n--- Testing median (odd length) ---\n";
    Vector v{1.0, 3.0, 2.0, 5.0, 4.0};
    double result = median(v);
    double expected = 3.0; // When sorted: [1,2,3,4,5], median is 3
    assert(approx_equal(result, expected) && "median (odd) failed");
    std::cout << "✓ median({1,3,2,5,4}) = " << result << "\n";
}

void test_median_even() {
    std::cout << "\n--- Testing median (even length) ---\n";
    Vector v{1.0, 2.0, 3.0, 4.0};
    double result = median(v);
    double expected = 2.5; // Sorted: [1,2,3,4], median is (2+3)/2 = 2.5
    assert(approx_equal(result, expected) && "median (even) failed");
    std::cout << "✓ median({1,2,3,4}) = " << result << "\n";
}

void test_quantile() {
    std::cout << "\n--- Testing quantile ---\n";
    Vector v{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    double q25 = quantile(v, 0.25);
    double q50 = quantile(v, 0.50);
    double q75 = quantile(v, 0.75);
    std::cout << "✓ quantile(0.25) = " << q25 << " (25th percentile)\n";
    std::cout << "✓ quantile(0.50) = " << q50 << " (median)\n";
    std::cout << "✓ quantile(0.75) = " << q75 << " (75th percentile)\n";
}

void test_mode() {
    std::cout << "\n--- Testing mode ---\n";
    Vector v{1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0};
    Vector result = mode(v);
    std::cout << "✓ mode({1,2,2,3,3,3,4}): ";
    for (double x : result) std::cout << x << " ";
    std::cout << "(most frequent: 3)\n";
    // 3 appears 3 times, which is the most, so it should be in the result
    assert(std::find(result.begin(), result.end(), 3.0) != result.end() && "mode failed");
}

void test_mode_empty() {
    std::cout << "\n--- Testing mode (empty vector) ---\n";
    Vector v;
    Vector result = mode(v);
    assert(result.empty() && "mode(empty) should return empty vector");
    std::cout << "✓ mode({}) returns empty vector\n";
}

void test_data_range() {
    std::cout << "\n--- Testing data_range ---\n";
    Vector v{1.0, 5.0, 3.0, 9.0, 2.0};
    double result = data_range(v);
    double expected = 9.0 - 1.0; // 8.0
    assert(approx_equal(result, expected) && "data_range failed");
    std::cout << "✓ data_range({1,5,3,9,2}) = " << result << " (max-min = 9-1)\n";
}

void test_de_mean() {
    std::cout << "\n--- Testing de_mean ---\n";
    Vector v{1.0, 2.0, 3.0};
    Vector result = de_mean(v);
    // mean is 2, so de_mean should be [-1, 0, 1]
    Vector expected{-1.0, 0.0, 1.0};
    assert(vectors_equal(result, expected) && "de_mean failed");
    std::cout << "✓ de_mean({1,2,3}) = [";
    for (size_t i = 0; i < result.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << result[i];
    }
    std::cout << "]\n";
}

void test_variance() {
    std::cout << "\n--- Testing variance ---\n";
    Vector v{2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0};
    double result = variance(v);
    std::cout << "✓ variance({2,4,4,4,5,5,7,9}) = " << result << "\n";
    assert(result > 0 && "variance should be positive");
}

void test_standard_deviation() {
    std::cout << "\n--- Testing standard_deviation ---\n";
    Vector v{2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0};
    double stdev = standard_deviation(v);
    double var = variance(v);
    double expected = std::sqrt(var);
    assert(approx_equal(stdev, expected) && "standard_deviation failed");
    std::cout << "✓ standard_deviation = " << stdev << " (sqrt of variance)\n";
}

void test_interquartile_range() {
    std::cout << "\n--- Testing interquartile_range ---\n";
    Vector v{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    double iqr = interquartile_range(v);
    std::cout << "✓ IQR = " << iqr << " (Q3 - Q1)\n";
    assert(iqr > 0 && "IQR should be positive");
}

void test_covariance() {
    std::cout << "\n--- Testing covariance ---\n";
    Vector x{1.0, 2.0, 3.0, 4.0, 5.0};
    Vector y{2.0, 4.0, 5.0, 4.0, 6.0};
    double cov = covariance(x, y);
    std::cout << "✓ covariance(x, y) = " << cov << "\n";
    // Positive correlation expected since y roughly increases with x
}

void test_correlation() {
    std::cout << "\n--- Testing correlation ---\n";
    Vector x{1.0, 2.0, 3.0, 4.0, 5.0};
    Vector y{2.0, 4.0, 6.0, 8.0, 10.0};
    double corr = correlation(x, y);
    std::cout << "✓ correlation(x, y) = " << corr << "\n";
    // Perfect positive correlation (y = 2*x), should be close to 1
    assert(corr > 0.99 && "perfect positive correlation should be ~1");
}

void test_correlation_zero_variance() {
    std::cout << "\n--- Testing correlation (zero variance) ---\n";
    Vector x{5.0, 5.0, 5.0};
    Vector y{1.0, 2.0, 3.0};
    double corr = correlation(x, y);
    assert(approx_equal(corr, 0.0) && "correlation with zero variance should return 0");
    std::cout << "✓ correlation with constant vector returns 0\n";
}

int main() {
    std::cout << "=============== Statistics Tests ===============\n";
    
    try {
        test_mean();
        test_median_odd();
        test_median_even();
        test_quantile();
        test_mode();
        test_mode_empty();
        test_data_range();
        test_de_mean();
        test_variance();
        test_standard_deviation();
        test_interquartile_range();
        test_covariance();
        test_correlation();
        test_correlation_zero_variance();
        
        std::cout << "\n=============== All Statistics Tests PASSED ✓ ===============\n";
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}
