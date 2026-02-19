#include <vector>
#include <functional>
#include <cassert>
#include <random>
#include "ds/linear_algebra.hpp"

namespace ds {

// ────────────────────────────────────────────────
// Basic gradient-related helper functions
// ────────────────────────────────────────────────

double sum_of_squares(const Vector& v) {
    return dot(v, v);
}

double difference_quotient(std::function<double(double)> f, double x, double h = 0.0001) {
    return (f(x + h) - f(x)) / h;
}

double square(double x) {
    return x * x;
}

// ────────────────────────────────────────────────
// Numerical gradient estimation (partial difference quotients)
// ────────────────────────────────────────────────

double partial_difference_quotient(
    std::function<double(const Vector&)> f,
    const Vector& v,
    size_t i,
    double h = 0.0001)
{
    Vector w = v;
    w[i] += h;

    return (f(w) - f(v)) / h;
}

Vector estimate_gradient(
    std::function<double(const Vector&)> f,
    const Vector& v,
    double h = 0.0001)
{
    Vector grad(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        grad[i] = partial_difference_quotient(f, v, i, h);
    }
    return grad;
}

// ────────────────────────────────────────────────
// Gradient descent steps
// ────────────────────────────────────────────────

Vector gradient_step(
    const Vector& v,
    const Vector& gradient,
    double step_size)
{
    assert(v.size() == gradient.size());

    // Note: step_size is usually negative when doing descent
    auto step = scalar_multiply(step_size, gradient);
    return add(v, step);
}

Vector sum_of_squares_gradient(const Vector& v) {
    return scalar_multiply(2.0, v);
}

// ────────────────────────────────────────────────
// Linear regression gradient
// ────────────────────────────────────────────────

Vector linear_gradient(double x, double y, const Vector& theta) {
    assert(theta.size() == 2);

    double slope     = theta[0];
    double intercept = theta[1];

    double predicted = slope * x + intercept;
    double error     = predicted - y;

    // gradient of squared error w.r.t. [slope, intercept]
    return Vector{
        2 * error * x,     // ∂/∂slope
        2 * error          // ∂/∂intercept
    };
}

// ────────────────────────────────────────────────
// Minibatch helper
// ────────────────────────────────────────────────

using DataPoint = std::pair<double, double>;  // (x, y)

template<typename T>
std::vector<std::vector<T>> minibatches(
    const std::vector<T>& dataset,
    size_t batch_size,
    bool shuffle = true)
{
    std::vector<std::vector<T>> batches;

    if (batch_size == 0 || dataset.empty()) {
        return batches;
    }

    std::vector<size_t> indices(dataset.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        indices[i] = i;
    }

    if (shuffle) {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);
    }

    for (size_t start = 0; start < dataset.size(); start += batch_size) {
        size_t end = std::min(start + batch_size, dataset.size());
        std::vector<T> batch;
        batch.reserve(end - start);

        for (size_t i = start; i < end; ++i) {
            batch.push_back(dataset[indices[i]]);
        }
        batches.push_back(std::move(batch));
    }

    return batches;
}

} 
