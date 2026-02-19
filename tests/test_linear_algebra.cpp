#include <iostream>
#include <cassert>
#include <cmath>
#include "ds/linear_algebra.hpp"

using namespace ds;

// Helper function to check floating point equality
bool approx_equal(double a, double b, double epsilon = 1e-9) {
    return std::abs(a - b) < epsilon;
}

// Helper to check vector equality
bool vectors_equal(const Vector& v1, const Vector& v2, double epsilon = 1e-9) {
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

// ============== Vector Operations Tests ==============

void test_vector_add() {
    std::cout << "\n--- Testing vector_add ---\n";
    Vector v{1.0, 2.0, 3.0};
    Vector w{4.0, 5.0, 6.0};
    Vector result = add(v, w);
    Vector expected{5.0, 7.0, 9.0};
    assert(vectors_equal(result, expected) && "vector_add failed");
    std::cout << "✓ vector_add passed\n";
}

void test_vector_subtract() {
    std::cout << "\n--- Testing vector_subtract ---\n";
    Vector v{5.0, 7.0, 9.0};
    Vector w{4.0, 5.0, 6.0};
    Vector result = subtract(v, w);
    Vector expected{1.0, 2.0, 3.0};
    assert(vectors_equal(result, expected) && "vector_subtract failed");
    std::cout << "✓ vector_subtract passed\n";
}

void test_scalar_multiply() {
    std::cout << "\n--- Testing scalar_multiply ---\n";
    Vector v{1.0, 2.0, 3.0};
    double c = 2.5;
    Vector result = scalar_multiply(c, v);
    Vector expected{2.5, 5.0, 7.5};
    assert(vectors_equal(result, expected) && "scalar_multiply failed");
    std::cout << "✓ scalar_multiply passed\n";
}

void test_vector_sum() {
    std::cout << "\n--- Testing vector_sum ---\n";
    std::vector<Vector> vectors{
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };
    Vector result = vector_sum(vectors);
    Vector expected{12.0, 15.0, 18.0};
    assert(vectors_equal(result, expected) && "vector_sum failed");
    std::cout << "✓ vector_sum passed\n";
}

void test_vector_mean() {
    std::cout << "\n--- Testing vector_mean ---\n";
    std::vector<Vector> vectors{
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };
    Vector result = vector_mean(vectors);
    Vector expected{4.0, 5.0, 6.0};
    assert(vectors_equal(result, expected) && "vector_mean failed");
    std::cout << "✓ vector_mean passed\n";
}

void test_dot_product() {
    std::cout << "\n--- Testing dot product ---\n";
    Vector v{1.0, 2.0, 3.0};
    Vector w{4.0, 5.0, 6.0};
    double result = dot(v, w);
    double expected = 1*4 + 2*5 + 3*6; // 4 + 10 + 18 = 32
    assert(approx_equal(result, expected) && "dot product failed");
    std::cout << "✓ dot product passed\n";
}

void test_sum_of_squares() {
    std::cout << "\n--- Testing sum_of_squares ---\n";
    Vector v{1.0, 2.0, 3.0};
    double result = sum_of_squares(v);
    double expected = 1*1 + 2*2 + 3*3; // 1 + 4 + 9 = 14
    assert(approx_equal(result, expected) && "sum_of_squares failed");
    std::cout << "✓ sum_of_squares passed\n";
}

void test_magnitude() {
    std::cout << "\n--- Testing magnitude ---\n";
    Vector v{3.0, 4.0};
    double result = magnitude(v);
    double expected = 5.0; // 3-4-5 triangle
    assert(approx_equal(result, expected) && "magnitude failed");
    std::cout << "✓ magnitude passed\n";
}

void test_distance() {
    std::cout << "\n--- Testing distance ---\n";
    Vector v{0.0, 0.0};
    Vector w{3.0, 4.0};
    double result = distance(v, w);
    double expected = 5.0;
    assert(approx_equal(result, expected) && "distance failed");
    std::cout << "✓ distance passed\n";
}

void test_squared_distance() {
    std::cout << "\n--- Testing squared_distance ---\n";
    Vector v{0.0, 0.0};
    Vector w{3.0, 4.0};
    double result = squared_distance(v, w);
    double expected = 25.0;
    assert(approx_equal(result, expected) && "squared_distance failed");
    std::cout << "✓ squared_distance passed\n";
}

// ============== Matrix Operations Tests ==============

void test_matrix_shape() {
    std::cout << "\n--- Testing matrix shape ---\n";
    Matrix A{
        {1, 2, 3},
        {4, 5, 6}
    };
    auto [rows, cols] = shape(A);
    assert(rows == 2 && cols == 3 && "matrix shape failed");
    std::cout << "✓ matrix shape passed\n";
}

void test_get_row() {
    std::cout << "\n--- Testing get_row ---\n";
    Matrix A{
        {1, 2, 3},
        {4, 5, 6}
    };
    Vector row = get_row(A, 1);
    Vector expected{4, 5, 6};
    assert(vectors_equal(row, expected) && "get_row failed");
    std::cout << "✓ get_row passed\n";
}

void test_get_column() {
    std::cout << "\n--- Testing get_column ---\n";
    Matrix A{
        {1, 2, 3},
        {4, 5, 6}
    };
    Vector col = get_column(A, 1);
    Vector expected{2, 5};
    assert(vectors_equal(col, expected) && "get_column failed");
    std::cout << "✓ get_column passed\n";
}

void test_identity_matrix() {
    std::cout << "\n--- Testing identity_matrix ---\n";
    Matrix I = identity_matrix(3);
    assert(I.size() == 3 && "identity matrix rows incorrect");
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            assert(approx_equal(I[i][j], expected) && "identity matrix element incorrect");
        }
    }
    std::cout << "✓ identity_matrix passed\n";
}

void test_make_matrix() {
    std::cout << "\n--- Testing make_matrix ---\n";
    Matrix M = make_matrix(2, 3, [](int i, int j) { return i + j; });
    assert(M.size() == 2 && M[0].size() == 3 && "make_matrix dimensions wrong");
    assert(approx_equal(M[0][0], 0) && "make_matrix[0][0] wrong");
    assert(approx_equal(M[1][2], 3) && "make_matrix[1][2] wrong");
    std::cout << "✓ make_matrix passed\n";
}

int main() {
    std::cout << "=============== Linear Algebra Tests ===============\n";
    
    try {
        // Vector tests
        test_vector_add();
        test_vector_subtract();
        test_scalar_multiply();
        test_vector_sum();
        test_vector_mean();
        test_dot_product();
        test_sum_of_squares();
        test_magnitude();
        test_distance();
        test_squared_distance();
        
        // Matrix tests
        test_matrix_shape();
        test_get_row();
        test_get_column();
        test_identity_matrix();
        test_make_matrix();
        
        std::cout << "\n=============== All Linear Algebra Tests PASSED ✓ ===============\n";
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}
