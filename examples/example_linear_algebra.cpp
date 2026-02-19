#include <iostream>
#include "ds/linear_algebra.hpp"
#include <vector>

int main() {
    using namespace ds;

    Vector v{1.0, 2.0, 3.0};
    Vector w{4.0, 5.0, 6.0};

    Vector s = add(v, w);
    std::cout << "v + w: ";
    for (double x : s) std::cout << x << " ";
    std::cout << '\n';

    std::cout << "dot(v, w): " << dot(v, w) << '\n';
    std::cout << "|v|: " << magnitude(v) << '\n';

    std::vector<Vector> vs{v, w};
    Vector mean = vector_mean(vs);
    std::cout << "mean(v, w): ";
    for (double x : mean) std::cout << x << " ";
    std::cout << '\n';

    Matrix I = identity_matrix(3);
    std::cout << "Identity 3x3:\n";
    for (const auto& row : I) {
        for (double x : row) std::cout << x << " ";
        std::cout << '\n';
    }

    return 0;
}
