#include "ds/probability.hpp"

#include <cmath>
#include <random>

namespace ds {

static std::mt19937 rng(std::random_device{}());

std::pair<double, double> normal_approximation_to_binomial(int n, double p) {
    double mu = p * n;
    double sigma = std::sqrt(p * (1 - p) * n);
    return {mu, sigma};
}

double normal_probability_above(double lo,
                                double mu,
                                double sigma) {
    return 1.0 - normal_cdf(lo, mu, sigma);
}

double normal_probability_between(double lo,
                                  double hi,
                                  double mu,
                                  double sigma) {
    return normal_cdf(hi, mu, sigma) -
           normal_cdf(lo, mu, sigma);
}

double normal_probability_outside(double lo,
                                  double hi,
                                  double mu,
                                  double sigma) {
    return 1.0 - normal_probability_between(lo, hi, mu, sigma);
}

double normal_upper_bound(double probability,
                          double mu,
                          double sigma) {
    return inverse_normal_cdf(probability, mu, sigma);
}

double normal_lower_bound(double probability,
                          double mu,
                          double sigma) {
    return inverse_normal_cdf(1.0 - probability, mu, sigma);
}

std::pair<double, double>
normal_two_sided_bounds(double probability,
                        double mu,
                        double sigma) {

    double tail_probability = (1.0 - probability) / 2.0;

    double upper_bound =
        normal_lower_bound(tail_probability, mu, sigma);

    double lower_bound =
        normal_upper_bound(tail_probability, mu, sigma);

    return {lower_bound, upper_bound};
}

double two_sided_p_value(double x,
                         double mu,
                         double sigma) {

    if (x >= mu)
        return 2.0 * normal_probability_above(x, mu, sigma);
    else
        return 2.0 * normal_cdf(x, mu, sigma);
}

std::vector<bool> run_experiment() {
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    std::vector<bool> flips(1000);

    for (int i = 0; i < 1000; ++i)
        flips[i] = (dist(rng) < 0.5);

    return flips;
}

bool reject_fairness(const std::vector<bool>& experiment) {

    int num_heads = 0;

    for (bool flip : experiment)
        if (flip) num_heads++;

    return (num_heads < 469 || num_heads > 531);
}
std::pair<double, double>
estimated_parameters(int N, int n) {
    double p = static_cast<double>(n) / N;
    double sigma = std::sqrt(p * (1 - p) / N);
    return {p, sigma};
}

double a_b_test_statistic(int N_A, int n_A,
                          int N_B, int n_B) {

    auto [p_A, sigma_A] = estimated_parameters(N_A, n_A);
    auto [p_B, sigma_B] = estimated_parameters(N_B, n_B);

    return (p_B - p_A) /
           std::sqrt(sigma_A * sigma_A +
                     sigma_B * sigma_B);
}

double B(double alpha, double beta) {
    return std::tgamma(alpha) *
           std::tgamma(beta) /
           std::tgamma(alpha + beta);
}

double beta_pdf(double x,
                double alpha,
                double beta) {

    if (x <= 0.0 || x >= 1.0)
        return 0.0;

    return std::pow(x, alpha - 1) *
           std::pow(1 - x, beta - 1) /
           B(alpha, beta);
}

} 
