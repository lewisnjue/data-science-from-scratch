#ifndef _HYPOTHESIS_TESTING_
#define _HYPOTHESIS_TESTING_

#include <utility>
#include <vector>

#include "ds/probability.hpp"

namespace ds {

std::pair<double, double> normal_approximation_to_binomial(int n, double p);

double normal_probability_above(double lo,
                                double mu = 0.0,
                                double sigma = 1.0);

double normal_probability_between(double lo,
                                  double hi,
                                  double mu = 0.0,
                                  double sigma = 1.0);

double normal_probability_outside(double lo,
                                  double hi,
                                  double mu = 0.0,
                                  double sigma = 1.0);

double normal_upper_bound(double probability,
                          double mu = 0.0,
                          double sigma = 1.0);

double normal_lower_bound(double probability,
                          double mu = 0.0,
                          double sigma = 1.0);

std::pair<double, double>
normal_two_sided_bounds(double probability,
                        double mu = 0.0,
                        double sigma = 1.0);

double two_sided_p_value(double x,
                         double mu = 0.0,
                         double sigma = 1.0);

std::vector<bool> run_experiment();
bool reject_fairness(const std::vector<bool>& experiment);

std::pair<double, double> estimated_parameters(int N, int n);

double a_b_test_statistic(int N_A, int n_A,
                          int N_B, int n_B);

double B(double alpha, double beta);
double beta_pdf(double x, double alpha, double beta);

} 

#endif
