#ifndef _PROBABILITY_
#define _PROBABILITY_

#include <random>

namespace ds {

double uniform_cdf(double x);

double normal_pdf(double x, double mu = 0.0, double sigma = 1.0);
double normal_cdf(double x, double mu = 0.0, double sigma = 1.0);

double inverse_normal_cdf(double p,
                          double mu = 0.0,
                          double sigma = 1.0,
                          double tolerance = 1e-5);
int bernoulli_trial(double p);

int binomial(int n, double p);

} // namespace ds

#endif
