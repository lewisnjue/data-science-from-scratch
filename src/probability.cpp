#include "ds/probability.hpp"
#include <cmath>
#include <cassert>

namespace ds {

static const double SQRT_TWO_PI = std::sqrt(2.0 * M_PI);

static std::mt19937 rng(std::random_device{}());

double uniform_cdf(double x) {
    if (x < 0.0)
        return 0.0;
    else if (x < 1.0)
        return x;
    else
        return 1.0;
}

double normal_pdf(double x, double mu, double sigma) {
    double exponent = -(std::pow(x - mu, 2)) / (2.0 * std::pow(sigma, 2));
    return std::exp(exponent) / (SQRT_TWO_PI * sigma);
}
double normal_cdf(double x, double mu, double sigma) {
    return (1.0 + std::erf((x - mu) / (std::sqrt(2.0) * sigma))) / 2.0;
}
double inverse_normal_cdf(double p,
                          double mu,
                          double sigma,
                          double tolerance) {

    // If not standard normal, rescale
    if (mu != 0.0 || sigma != 1.0) {
        return mu + sigma *
               inverse_normal_cdf(p, 0.0, 1.0, tolerance);
    }

    double low_z = -10.0;
    double hi_z  =  10.0;
    double mid_z = 0.0;

    while (hi_z - low_z > tolerance) {
        mid_z = (low_z + hi_z) / 2.0;
        double mid_p = normal_cdf(mid_z);

        if (mid_p < p)
            low_z = mid_z;
        else
            hi_z = mid_z;
    }

    return mid_z;
}

int bernoulli_trial(double p) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return (dist(rng) < p) ? 1 : 0;
}
int binomial(int n, double p) {
    int sum = 0;
    for (int i = 0; i < n; ++i) {
        sum += bernoulli_trial(p);
    }
    return sum;
}

} 
