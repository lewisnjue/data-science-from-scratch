#if !defined(__GRADIENT__)
#define __GRADIENT__

#include <vector>
#include <functional>
#include <utility>
#include "ds/linear_algebra.hpp"

namespace ds {

// ────────────────────────────────────────────────
// Basic gradient-related helper functions
// ────────────────────────────────────────────────

/// Compute the sum of squares of a vector
/// @param v Input vector
/// @return dot(v, v)
double sum_of_squares(const Vector& v);

/// Compute the numerical derivative using the difference quotient
/// @param f The function to differentiate
/// @param x The point at which to compute the derivative
/// @param h The step size (default 0.0001)
/// @return Approximate derivative at x
double difference_quotient(std::function<double(double)> f, double x, double h = 0.0001);

/// Compute the square of a number
/// @param x Input value
/// @return x * x
double square(double x);

// ────────────────────────────────────────────────
// Numerical gradient estimation (partial difference quotients)
// ────────────────────────────────────────────────

/// Compute the partial difference quotient (numerical partial derivative)
/// @param f The multivariate function
/// @param v The point at which to compute the partial derivative
/// @param i The index of the variable to differentiate with respect to
/// @param h The step size (default 0.0001)
/// @return Approximate partial derivative ∂f/∂v[i]
double partial_difference_quotient(
    std::function<double(const Vector&)> f,
    const Vector& v,
    size_t i,
    double h = 0.0001);

/// Estimate the gradient of a multivariate function at a point
/// @param f The function to differentiate
/// @param v The point at which to compute the gradient
/// @param h The step size (default 0.0001)
/// @return Vector of partial derivatives (numerical gradient)
Vector estimate_gradient(
    std::function<double(const Vector&)> f,
    const Vector& v,
    double h = 0.0001);

// ────────────────────────────────────────────────
// Gradient descent steps
// ────────────────────────────────────────────────

/// Perform one step of gradient descent
/// @param v The current point
/// @param gradient The gradient at v
/// @param step_size The step size (usually negative for descent)
/// @return Updated point after one gradient step
Vector gradient_step(
    const Vector& v,
    const Vector& gradient,
    double step_size);

/// Compute the gradient of the sum of squares function
/// @param v Input vector
/// @return 2*v (analytical gradient of ||v||²)
Vector sum_of_squares_gradient(const Vector& v);

// ────────────────────────────────────────────────
// Linear regression gradient
// ────────────────────────────────────────────────

/// Compute the gradient of squared error for linear regression
/// @param x The input feature value
/// @param y The actual target value
/// @param theta The model parameters [slope, intercept]
/// @return Gradient vector with respect to [slope, intercept]
Vector linear_gradient(double x, double y, const Vector& theta);

// ────────────────────────────────────────────────
// Minibatch helper
// ────────────────────────────────────────────────

/// Type alias for a data point (x, y pair)
using DataPoint = std::pair<double, double>;

/// Create minibatches from a dataset
/// @tparam T The type of elements in the dataset
/// @param dataset The full dataset to split into batches
/// @param batch_size The size of each batch
/// @param shuffle Whether to randomly shuffle the data before batching (default true)
/// @return A vector of minibatches
template<typename T>
std::vector<std::vector<T>> minibatches(
    const std::vector<T>& dataset,
    size_t batch_size,
    bool shuffle = true);

} // namespace ds

#endif // __GRADIENT__           