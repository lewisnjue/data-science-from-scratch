#include "ds/linear_algebra.hpp"
#include <cassert>
#include <cmath>

// ---------------- Vector ----------------
namespace ds {
Vector add(const Vector& v, const Vector& w) {
    assert(v.size() == w.size());
    Vector result(v.size());

    for (size_t i = 0; i < v.size(); ++i)
        result[i] = v[i] + w[i];

    return result;
}

Vector subtract(const Vector& v, const Vector& w) {
    assert(v.size() == w.size());
    Vector result(v.size());

    for (size_t i = 0; i < v.size(); ++i)
        result[i] = v[i] - w[i];

    return result;
}

Vector scalar_multiply(double c, const Vector& v) {
    Vector result(v.size());

    for (size_t i = 0; i < v.size(); ++i)
        result[i] = c * v[i];

    return result;
}

Vector vector_sum(const std::vector<Vector>& vectors) {
    assert(!vectors.empty());

    size_t n = vectors[0].size();
    Vector result(n, 0.0);

    for (const auto& v : vectors) {
        assert(v.size() == n);
        for (size_t i = 0; i < n; ++i)
            result[i] += v[i];
    }

    return result;
}

Vector vector_mean(const std::vector<Vector>& vectors) {
    return scalar_multiply(1.0 / vectors.size(), vector_sum(vectors));
}

double dot(const Vector& v, const Vector& w) {
    assert(v.size() == w.size());

    double sum = 0.0;
    for (size_t i = 0; i < v.size(); ++i)
        sum += v[i] * w[i];

    return sum;
}

double sum_of_squares(const Vector& v) {
    return dot(v, v);
}

double magnitude(const Vector& v) {
    return std::sqrt(sum_of_squares(v));
}

double squared_distance(const Vector& v, const Vector& w) {
    return sum_of_squares(subtract(v, w));
}

double distance(const Vector& v, const Vector& w) {
    return magnitude(subtract(v, w));
}

// ---------------- Matrix ----------------

std::pair<int, int> shape(const Matrix& A) {
    int rows = A.size();
    int cols = rows > 0 ? A[0].size() : 0;
    return {rows, cols};
}

Vector get_row(const Matrix& A, int i) {
    return A[i];
}

Vector get_column(const Matrix& A, int j) {
    Vector column(A.size());

    for (size_t i = 0; i < A.size(); ++i)
        column[i] = A[i][j];

    return column;
}

Matrix make_matrix(
    int num_rows,
    int num_cols,
    std::function<double(int, int)> entry_fn) {

    Matrix result(num_rows, Vector(num_cols));

    for (int i = 0; i < num_rows; ++i)
        for (int j = 0; j < num_cols; ++j)
            result[i][j] = entry_fn(i, j);

    return result;
}

Matrix identity_matrix(int n) {
    return make_matrix(n, n,
        [](int i, int j) {
            return (i == j) ? 1.0 : 0.0;
        });
}
}